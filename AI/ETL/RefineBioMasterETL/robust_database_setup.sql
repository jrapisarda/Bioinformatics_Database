-- ===================================================================
-- Production-Grade Bioinformatics Data Warehouse Database Setup
-- Creates all tables, indexes, and stored procedures with MERGE operations
-- Supports scalable study code patterns (SRP*, EX-AUR-1004, etc.)
-- ===================================================================

USE BioinformaticsWarehouse;
GO

-- ===================================================================
-- SCHEMAS
-- ===================================================================

-- Create staging schema for temporary data loading
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'staging')
    EXEC('CREATE SCHEMA staging')
GO

-- Create audit schema for logging and monitoring
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'audit')
    EXEC('CREATE SCHEMA audit')
GO

-- ===================================================================
-- AUDIT AND LOGGING TABLES
-- ===================================================================

-- ETL Execution Log
CREATE TABLE audit.etl_execution_log (
    execution_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    study_code VARCHAR(50) NOT NULL,
    batch_id VARCHAR(50) NOT NULL,
    pipeline_version VARCHAR(20) DEFAULT '1.0.0',
    start_time DATETIME2 NOT NULL,
    end_time DATETIME2 NULL,
    duration_seconds INT NULL,
    status VARCHAR(20) NOT NULL, -- 'IN_PROGRESS', 'SUCCESS', 'FAILED', 'WARNING'
    records_processed INT DEFAULT 0,
    records_inserted INT DEFAULT 0,
    records_updated INT DEFAULT 0,
    records_failed INT DEFAULT 0,
    error_message NVARCHAR(MAX) NULL,
    validation_score DECIMAL(5,2) NULL,
    performance_metrics NVARCHAR(MAX) NULL,
    created_date DATETIME2 DEFAULT GETDATE()
);

-- Data Quality Log
CREATE TABLE audit.data_quality_log (
    quality_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    execution_id BIGINT NOT NULL,
    study_code VARCHAR(50) NOT NULL,
    check_type VARCHAR(100) NOT NULL,
    check_result VARCHAR(20) NOT NULL, -- 'PASS', 'WARNING', 'ERROR'
    quality_metric VARCHAR(200) NULL,
    quality_value DECIMAL(10,4) NULL,
    details NVARCHAR(MAX) NULL,
    timestamp DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (execution_id) REFERENCES audit.etl_execution_log(execution_id)
);

-- Performance Monitoring Log
CREATE TABLE audit.performance_log (
    performance_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    execution_id BIGINT NOT NULL,
    study_code VARCHAR(50) NOT NULL,
    operation_type VARCHAR(50) NOT NULL, -- 'EXTRACTION', 'VALIDATION', 'TRANSFORMATION', 'LOADING'
    operation_name VARCHAR(100) NOT NULL,
    start_time DATETIME2 NOT NULL,
    end_time DATETIME2 NOT NULL,
    duration_ms INT NOT NULL,
    memory_usage_mb INT NULL,
    records_processed INT NULL,
    throughput_records_per_second DECIMAL(10,2) NULL,
    timestamp DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (execution_id) REFERENCES audit.etl_execution_log(execution_id)
);

-- ===================================================================
-- DIMENSION TABLES
-- ===================================================================

-- Study Dimension
CREATE TABLE dim_study (
    study_key INT IDENTITY(1,1) PRIMARY KEY,
    accession_code VARCHAR(50) NOT NULL UNIQUE,
    title VARCHAR(500) NULL,
    description NVARCHAR(MAX) NULL,
    technology VARCHAR(100) NULL,
    organism VARCHAR(100) NULL,
    has_publication BIT DEFAULT 0,
    pubmed_id VARCHAR(50) NULL,
    publication_title VARCHAR(500) NULL,
    source_first_published DATE NULL,
    source_last_modified DATE NULL,
    source_url VARCHAR(500) NULL,
    source_archive_url VARCHAR(500) NULL,
    has_public_raw_data BIT DEFAULT 0,
    created_date DATETIME2 DEFAULT GETDATE(),
    updated_date DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT CK_study_technology CHECK (technology IN ('RNA-SEQ', 'MICROARRAY', 'OTHER')),
    INDEX IX_study_accession UNIQUE (accession_code),
    INDEX IX_study_organism (organism),
    INDEX IX_study_created_date (created_date)
);

-- Platform Dimension
CREATE TABLE dim_platform (
    platform_key INT IDENTITY(1,1) PRIMARY KEY,
    platform_name VARCHAR(200) NOT NULL UNIQUE,
    platform_type VARCHAR(100) NULL,
    platform_technology VARCHAR(100) NULL,
    manufacturer VARCHAR(100) NULL,
    processor_name VARCHAR(100) NULL,
    processor_version VARCHAR(50) NULL,
    processor_id INT NULL,
    annotation_package VARCHAR(100) NULL,
    created_date DATETIME2 DEFAULT GETDATE(),
    updated_date DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT CK_platform_type CHECK (platform_type IN ('ILLUMINA', 'AFFYMETRIX', 'AGILENT', 'OTHER')),
    INDEX IX_platform_name UNIQUE (platform_name),
    INDEX IX_platform_type (platform_type)
);

-- Illness/Condition Dimension
CREATE TABLE dim_illness (
    illness_key INT IDENTITY(1,1) PRIMARY KEY,
    illness_type VARCHAR(50) NOT NULL UNIQUE,
    illness_description VARCHAR(500) NULL,
    illness_category VARCHAR(100) NULL,
    created_date DATETIME2 DEFAULT GETDATE(),
    updated_date DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT CK_illness_type CHECK (illness_type IN ('control', 'sepsis', 'infection', 'disease', 'unknown')),
    INDEX IX_illness_type UNIQUE (illness_type)
);

-- Sample Dimension
CREATE TABLE dim_sample (
    sample_key INT IDENTITY(1,1) PRIMARY KEY,
    refinebio_accession_code VARCHAR(50) NOT NULL UNIQUE,
    experiment_accession VARCHAR(50) NULL,
    refinebio_title VARCHAR(500) NULL,
    refinebio_organism VARCHAR(100) NULL,
    refinebio_processed BIT DEFAULT 0,
    refinebio_source_database VARCHAR(50) NULL,
    refinebio_source_url VARCHAR(500) NULL,
    refinebio_has_raw BIT DEFAULT 0,
    refinebio_platform VARCHAR(200) NULL,
    sample_classification VARCHAR(100) NULL,
    sample_number INT NULL,
    sample_description VARCHAR(500) NULL,
    age VARCHAR(50) NULL,
    sex VARCHAR(20) NULL,
    tissue VARCHAR(100) NULL,
    cell_line VARCHAR(100) NULL,
    treatment VARCHAR(200) NULL,
    created_date DATETIME2 DEFAULT GETDATE(),
    updated_date DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT CK_sample_accession CHECK (refinebio_accession_code LIKE 'SRR%' OR refinebio_accession_code LIKE 'ERR%'),
    INDEX IX_sample_accession UNIQUE (refinebio_accession_code),
    INDEX IX_sample_experiment (experiment_accession),
    INDEX IX_sample_organism (refinebio_organism),
    INDEX IX_sample_classification (sample_classification),
    INDEX IX_sample_created_date (created_date)
);

-- Gene Dimension
CREATE TABLE dim_gene (
    gene_key INT IDENTITY(1,1) PRIMARY KEY,
    gene_symbol VARCHAR(100) NOT NULL UNIQUE,
    gene_description VARCHAR(1000) NULL,
    gene_type VARCHAR(100) DEFAULT 'protein_coding',
    chromosome VARCHAR(10) NULL,
    gene_start INT NULL,
    gene_end INT NULL,
    gene_length INT NULL,
    strand CHAR(1) NULL,
    ensembl_id VARCHAR(50) NULL,
    entrez_id VARCHAR(20) NULL,
    hgnc_id VARCHAR(20) NULL,
    created_date DATETIME2 DEFAULT GETDATE(),
    updated_date DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT CK_gene_type CHECK (gene_type IN ('protein_coding', 'lncRNA', 'miRNA', 'rRNA', 'tRNA', 'pseudogene', 'other')),
    CONSTRAINT CK_gene_strand CHECK (strand IN ('+', '-', 'N') OR strand IS NULL),
    INDEX IX_gene_symbol UNIQUE (gene_symbol),
    INDEX IX_gene_ensembl UNIQUE (ensembl_id),
    INDEX IX_gene_chromosome (chromosome),
    INDEX IX_gene_type (gene_type)
);

-- ===================================================================
-- FACT TABLE
-- ===================================================================

-- Gene Expression Fact Table
CREATE TABLE fact_gene_expression (
    fact_key BIGINT IDENTITY(1,1) PRIMARY KEY,
    sample_key INT NOT NULL,
    gene_key INT NOT NULL,
    expression_value DECIMAL(15,6) NOT NULL,
    normalized_expression_value DECIMAL(15,6) NULL,
    z_score DECIMAL(10,6) NULL,
    quantile_normalized BIT DEFAULT 0,
    is_outlier BIT DEFAULT 0,
    etl_batch_id VARCHAR(50) NOT NULL,
    etl_processed_at DATETIME2 DEFAULT GETDATE(),
    data_source VARCHAR(100) DEFAULT 'refine.bio',
    
    -- Foreign key constraints
    FOREIGN KEY (sample_key) REFERENCES dim_sample(sample_key),
    FOREIGN KEY (gene_key) REFERENCES dim_gene(gene_key),
    
    -- Unique constraint to prevent duplicates
    CONSTRAINT UK_expression UNIQUE (sample_key, gene_key),
    
    -- Performance indexes
    INDEX IX_expression_sample_gene (sample_key, gene_key),
    INDEX IX_expression_gene_sample (gene_key, sample_key),
    INDEX IX_expression_value (expression_value),
    INDEX IX_expression_batch (etl_batch_id),
    INDEX IX_expression_processed (etl_processed_at)
);

-- ===================================================================
-- INDEXES FOR PERFORMANCE
-- ===================================================================

-- Additional performance indexes
CREATE INDEX IX_fact_expression_sample ON fact_gene_expression(sample_key) 
INCLUDE (gene_key, expression_value);

CREATE INDEX IX_fact_expression_gene ON fact_gene_expression(gene_key) 
INCLUDE (sample_key, expression_value);

CREATE INDEX IX_fact_expression_high_values ON fact_gene_expression(expression_value) 
WHERE expression_value > 1000;

-- ===================================================================
-- STORED PROCEDURES FOR ETL OPERATIONS
-- ===================================================================

-- ETL Execution Logging Procedure
CREATE OR ALTER PROCEDURE audit.sp_log_etl_execution_start
    @study_code VARCHAR(50),
    @batch_id VARCHAR(50) OUTPUT,
    @execution_id BIGINT OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    
    SET @batch_id = 'ETL_' + @study_code + '_' + FORMAT(GETDATE(), 'yyyyMMdd_HHmmss');
    
    INSERT INTO audit.etl_execution_log (study_code, batch_id, start_time, status)
    VALUES (@study_code, @batch_id, GETDATE(), 'IN_PROGRESS');
    
    SET @execution_id = SCOPE_IDENTITY();
    
    RETURN @execution_id;
END;
GO

CREATE OR ALTER PROCEDURE audit.sp_log_etl_execution_end
    @execution_id BIGINT,
    @status VARCHAR(20),
    @records_processed INT = 0,
    @records_inserted INT = 0,
    @records_updated INT = 0,
    @records_failed INT = 0,
    @error_message NVARCHAR(MAX) = NULL,
    @validation_score DECIMAL(5,2) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    UPDATE audit.etl_execution_log 
    SET 
        end_time = GETDATE(),
        duration_seconds = DATEDIFF(SECOND, start_time, GETDATE()),
        status = @status,
        records_processed = @records_processed,
        records_inserted = @records_inserted,
        records_updated = @records_updated,
        records_failed = @records_failed,
        error_message = @error_message,
        validation_score = @validation_score
    WHERE execution_id = @execution_id;
END;
GO

-- Data Quality Logging Procedure
CREATE OR ALTER PROCEDURE audit.sp_log_data_quality
    @execution_id BIGINT,
    @study_code VARCHAR(50),
    @check_type VARCHAR(100),
    @check_result VARCHAR(20),
    @quality_metric VARCHAR(200) = NULL,
    @quality_value DECIMAL(10,4) = NULL,
    @details NVARCHAR(MAX) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO audit.data_quality_log (execution_id, study_code, check_type, check_result, 
                                       quality_metric, quality_value, details)
    VALUES (@execution_id, @study_code, @check_type, @check_result, 
            @quality_metric, @quality_value, @details);
END;
GO

-- Performance Monitoring Procedure
CREATE OR ALTER PROCEDURE audit.sp_log_performance
    @execution_id BIGINT,
    @study_code VARCHAR(50),
    @operation_type VARCHAR(50),
    @operation_name VARCHAR(100),
    @start_time DATETIME2,
    @end_time DATETIME2,
    @memory_usage_mb INT = NULL,
    @records_processed INT = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @duration_ms INT = DATEDIFF(MILLISECOND, @start_time, @end_time);
    DECLARE @throughput DECIMAL(10,2) = NULL;
    
    IF @records_processed IS NOT NULL AND @duration_ms > 0
        SET @throughput = CAST(@records_processed AS DECIMAL) / (@duration_ms / 1000.0);
    
    INSERT INTO audit.performance_log (execution_id, study_code, operation_type, operation_name,
                                      start_time, end_time, duration_ms, memory_usage_mb,
                                      records_processed, throughput_records_per_second)
    VALUES (@execution_id, @study_code, @operation_type, @operation_name,
            @start_time, @end_time, @duration_ms, @memory_usage_mb,
            @records_processed, @throughput);
END;
GO

-- ===================================================================
-- MERGE PROCEDURES FOR DIMENSION LOADING
-- ===================================================================

CREATE OR ALTER PROCEDURE sp_merge_dim_study
    @staging_table_name SYSNAME,
    @execution_id BIGINT
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    
    SET @sql = N'
    MERGE dim_study AS target
    USING ' + QUOTENAME(@staging_table_name) + ' AS source
    ON target.accession_code = source.accession_code
    
    WHEN MATCHED THEN
        UPDATE SET 
            title = source.title,
            description = source.description,
            technology = source.technology,
            organism = source.organism,
            has_publication = source.has_publication,
            pubmed_id = source.pubmed_id,
            publication_title = source.publication_title,
            source_first_published = source.source_first_published,
            source_last_modified = source.source_last_modified,
            updated_date = GETDATE()
    
    WHEN NOT MATCHED THEN
        INSERT (accession_code, title, description, technology, organism, 
               has_publication, pubmed_id, publication_title, 
               source_first_published, source_last_modified)
        VALUES (source.accession_code, source.title, source.description, 
               source.technology, source.organism, source.has_publication, 
               source.pubmed_id, source.publication_title, 
               source.source_first_published, source.source_last_modified);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, 'SYSTEM', 'DIMENSION_LOAD', 'dim_study',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

CREATE OR ALTER PROCEDURE sp_merge_dim_platform
    @staging_table_name SYSNAME,
    @execution_id BIGINT
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    
    SET @sql = N'
    MERGE dim_platform AS target
    USING ' + QUOTENAME(@staging_table_name) + ' AS source
    ON target.platform_name = source.platform_name
    
    WHEN MATCHED THEN
        UPDATE SET 
            platform_type = source.platform_type,
            processor_name = source.processor_name,
            processor_version = source.processor_version,
            processor_id = source.processor_id,
            updated_date = GETDATE()
    
    WHEN NOT MATCHED THEN
        INSERT (platform_name, platform_type, processor_name, processor_version, processor_id)
        VALUES (source.platform_name, source.platform_type, source.processor_name, 
               source.processor_version, source.processor_id);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, 'SYSTEM', 'DIMENSION_LOAD', 'dim_platform',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

CREATE OR ALTER PROCEDURE sp_merge_dim_illness
    @staging_table_name SYSNAME,
    @execution_id BIGINT
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    
    SET @sql = N'
    MERGE dim_illness AS target
    USING ' + QUOTENAME(@staging_table_name) + ' AS source
    ON target.illness_type = source.illness_type
    
    WHEN MATCHED THEN
        UPDATE SET 
            illness_description = source.illness_description,
            updated_date = GETDATE()
    
    WHEN NOT MATCHED THEN
        INSERT (illness_type, illness_description)
        VALUES (source.illness_type, source.illness_description);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, 'SYSTEM', 'DIMENSION_LOAD', 'dim_illness',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

CREATE OR ALTER PROCEDURE sp_merge_dim_sample
    @staging_table_name SYSNAME,
    @execution_id BIGINT
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    
    SET @sql = N'
    MERGE dim_sample AS target
    USING ' + QUOTENAME(@staging_table_name) + ' AS source
    ON target.refinebio_accession_code = source.refinebio_accession_code
    
    WHEN MATCHED THEN
        UPDATE SET 
            experiment_accession = source.experiment_accession,
            refinebio_title = source.refinebio_title,
            refinebio_organism = source.refinebio_organism,
            refinebio_processed = source.refinebio_processed,
            refinebio_source_database = source.refinebio_source_database,
            sample_classification = source.sample_classification,
            sample_number = source.sample_number,
            updated_date = GETDATE()
    
    WHEN NOT MATCHED THEN
        INSERT (refinebio_accession_code, experiment_accession, refinebio_title, 
               refinebio_organism, refinebio_processed, refinebio_source_database,
               sample_classification, sample_number)
        VALUES (source.refinebio_accession_code, source.experiment_accession, 
               source.refinebio_title, source.refinebio_organism, source.refinebio_processed,
               source.refinebio_source_database, source.sample_classification, 
               source.sample_number);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, 'SYSTEM', 'DIMENSION_LOAD', 'dim_sample',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

CREATE OR ALTER PROCEDURE sp_merge_dim_gene
    @staging_table_name SYSNAME,
    @execution_id BIGINT
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    
    SET @sql = N'
    MERGE dim_gene AS target
    USING ' + QUOTENAME(@staging_table_name) + ' AS source
    ON target.gene_symbol = source.gene_symbol
    
    WHEN MATCHED THEN
        UPDATE SET 
            gene_description = source.gene_description,
            gene_type = source.gene_type,
            chromosome = source.chromosome,
            gene_length = source.gene_length,
            strand = source.strand,
            updated_date = GETDATE()
    
    WHEN NOT MATCHED THEN
        INSERT (gene_symbol, gene_description, gene_type, chromosome, gene_length, strand)
        VALUES (source.gene_symbol, source.gene_description, source.gene_type,
               source.chromosome, source.gene_length, source.strand);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, 'SYSTEM', 'DIMENSION_LOAD', 'dim_gene',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

-- ===================================================================
-- MERGE PROCEDURE FOR FACT TABLE
-- ===================================================================

CREATE OR ALTER PROCEDURE sp_merge_fact_expression
    @staging_table_name SYSNAME,
    @execution_id BIGINT,
    @study_code VARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @records_affected INT;
    DECLARE @batch_id VARCHAR(50);
    
    -- Get batch ID from staging table
    SET @sql = N'SELECT TOP 1 @batch_id_out = etl_batch_id FROM ' + QUOTENAME(@staging_table_name);
    EXEC sp_executesql @sql, N'@batch_id_out VARCHAR(50) OUTPUT', @batch_id_out = @batch_id OUTPUT;
    
    -- Perform MERGE with deduplication logic
    SET @sql = N'
    MERGE fact_gene_expression AS target
    USING (
        SELECT 
            s.sample_key, 
            g.gene_key, 
            source.expression_value,
            source.etl_batch_id,
            source.etl_processed_at,
            ROW_NUMBER() OVER (
                PARTITION BY s.sample_key, g.gene_key 
                ORDER BY source.etl_processed_at DESC
            ) as rn
        FROM ' + QUOTENAME(@staging_table_name) + ' source
        INNER JOIN dim_sample s ON source.refinebio_accession_code = s.refinebio_accession_code
        INNER JOIN dim_gene g ON source.gene_symbol = g.gene_symbol
    ) AS source
    ON target.sample_key = source.sample_key 
    AND target.gene_key = source.gene_key
    AND source.rn = 1  -- Only take the most recent duplicate
    
    WHEN MATCHED THEN
        UPDATE SET 
            expression_value = source.expression_value,
            etl_batch_id = source.etl_batch_id,
            etl_processed_at = source.etl_processed_at
    
    WHEN NOT MATCHED THEN
        INSERT (sample_key, gene_key, expression_value, etl_batch_id, etl_processed_at)
        VALUES (source.sample_key, source.gene_key, source.expression_value,
               source.etl_batch_id, source.etl_processed_at);'
    
    EXEC sp_executesql @sql;
    SET @records_affected = @@ROWCOUNT;
    
    -- Log the operation
    EXEC audit.sp_log_performance @execution_id, @study_code, 'FACT_LOAD', 'fact_gene_expression',
         GETDATE(), GETDATE(), NULL, @records_affected;
    
    RETURN @records_affected;
END;
GO

-- ===================================================================
-- UTILITY PROCEDURES
-- ===================================================================

-- Procedure to get ETL statistics
CREATE OR ALTER PROCEDURE sp_get_etl_statistics
    @study_code VARCHAR(50) = NULL,
    @start_date DATE = NULL,
    @end_date DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    SELECT 
        e.execution_id,
        e.study_code,
        e.batch_id,
        e.start_time,
        e.end_time,
        e.duration_seconds,
        e.status,
        e.records_processed,
        e.records_inserted,
        e.records_updated,
        e.records_failed,
        e.validation_score,
        COUNT(DISTINCT q.quality_id) as quality_checks,
        SUM(CASE WHEN q.check_result = 'ERROR' THEN 1 ELSE 0 END) as quality_errors,
        SUM(CASE WHEN q.check_result = 'WARNING' THEN 1 ELSE 0 END) as quality_warnings
    FROM audit.etl_execution_log e
    LEFT JOIN audit.data_quality_log q ON e.execution_id = q.execution_id
    WHERE 
        (@study_code IS NULL OR e.study_code = @study_code)
        AND (@start_date IS NULL OR CAST(e.start_time AS DATE) >= @start_date)
        AND (@end_date IS NULL OR CAST(e.start_time AS DATE) <= @end_date)
    GROUP BY e.execution_id, e.study_code, e.batch_id, e.start_time, e.end_time,
             e.duration_seconds, e.status, e.records_processed, e.records_inserted,
             e.records_updated, e.records_failed, e.validation_score
    ORDER BY e.start_time DESC;
END;
GO

-- Procedure to clean up old ETL logs
CREATE OR ALTER PROCEDURE sp_cleanup_etl_logs
    @retention_days INT = 90
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @cutoff_date DATETIME2 = DATEADD(DAY, -@retention_days, GETDATE());
    
    -- Delete old performance logs
    DELETE FROM audit.performance_log 
    WHERE timestamp < @cutoff_date;
    
    -- Delete old quality logs
    DELETE FROM audit.data_quality_log 
    WHERE timestamp < @cutoff_date;
    
    -- Delete old execution logs
    DELETE FROM audit.etl_execution_log 
    WHERE start_time < @cutoff_date;
    
    PRINT 'ETL log cleanup completed for data older than ' + CAST(@retention_days AS VARCHAR) + ' days';
END;
GO

-- ===================================================================
-- SAMPLE DATA FOR TESTING
-- ===================================================================

-- Insert default illness types
INSERT INTO dim_illness (illness_type, illness_description) VALUES
('control', 'Control/Healthy sample'),
('sepsis', 'Sepsis/Infected sample'),
('infection', 'General infection sample'),
('disease', 'Disease state sample'),
('unknown', 'Unknown/Unclassified sample');

-- Insert default platform
INSERT INTO dim_platform (platform_name, platform_type, manufacturer) VALUES
('Unknown', 'OTHER', 'Unknown');

PRINT 'Database setup completed successfully';
GO
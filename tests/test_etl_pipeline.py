import pandas as pd

from AI.ETL.RefineBioMasterETL.robust_data_transformer import DataTransformer


def test_transform_all_data_handles_aggregated_json_metadata():
    study_code = "SRP123456"
    sample_id = "SRR000001"

    aggregated_metadata = {
        "experiments": {
            study_code: {
                "accession_code": study_code,
                "title": "Aggregated Study",
                "description": "Study sourced from aggregated metadata payload",
                "technology": "RNA-SEQ",
                "organism": "Homo sapiens",
                "has_publication": False,
                "samples": {
                    sample_id: {
                        "platform_name": "Illumina HiSeq",
                        "processor_name": "refinebio",
                        "processor_version": "1.0",
                        "processor_id": 101,
                    }
                },
            }
        }
    }

    tsv_metadata = pd.DataFrame(
        [
            {
                "refinebio_accession_code": sample_id,
                "experiment_accession_code": study_code,
                "refinebio_title": "control sample 1",
                "refinebio_organism": "Homo sapiens",
                "refinebio_processed": True,
                "refinebio_source_database": "SRA",
            }
        ]
    )

    expression_data = pd.DataFrame(
        [[1.23]], index=["GENE1"], columns=[sample_id]
    )

    extracted_data = {
        "study_code": study_code,
        "json_metadata": aggregated_metadata,
        "tsv_metadata": tsv_metadata,
        "expression_data": expression_data,
    }

    transformer = DataTransformer(config=object())
    transformed = transformer.transform_all_data(extracted_data)

    dim_study = transformed["dimensions"]["dim_study"]
    assert dim_study.loc[0, "accession_code"] == study_code

    dim_samples = transformed["dimensions"]["dim_samples"]
    assert sample_id in dim_samples["refinebio_accession_code"].values

import polars as pl


def format_outputs(predictions, sample_predictions):
    """ make kaggle compatible predictions for submission """
    all_predictions = pl.concat(predictions, how='vertical'
                       ).select('row_id', 'prediction'
                       ).rename({'prediction': 'target'}
                       ).to_pandas()
 
    return sample_predictions.merge(
                                all_predictions[['row_id', 'target']], 
                                on='row_id', how='left', suffixes=['_old', '']
                            ).drop(
                                columns='target_old'
                            ).fillna(0.0)

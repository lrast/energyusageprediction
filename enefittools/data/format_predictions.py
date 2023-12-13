def format_outputs(full_predictions, sample_predictions):
    """ make kaggle compatible predictions for submission """
    return sample_predictions.merge(full_predictions[['row_id', 'target']], 
                                    on='row_id', how='left', suffixes=['_old', '']
                            ).drop(columns='target_old'
                            ).fillna(0.0)

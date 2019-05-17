# deeplearning

## Training##
- python main.py --training_settings_name {folder name to differentiate  models,logs } --feature_extractor 2 --activation 2 --optimizer 1 --add_dropout 1 --early_stopping 0 --use_batch_normalisation 1 --epochs {no. of epochs} --initial_weights_path {path to weights}

## Testing ##
- python main.py --training_settings_name {folder name to store accuracies} --feature_extractor 2 --activation 2 --optimizer 1 --add_dropout 1 --early_stopping 0 --use_batch_normalisation 1 --initial_weights_path {path to weights} --is_training 0

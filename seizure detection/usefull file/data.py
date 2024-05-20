class Data:
    TRAIN_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
    TEST_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/test.csv'


    @staticmethod
    def load_train_info(num_samples = None):
        train_info = (
            pd.read_csv(Data.TRAIN_PATH)
            .drop(columns = [
                'expert_consensus',
                'eeg_sub_id',
                'spectrogram_sub_id',
                'patient_id',
                'label_id'
            ])
        )
        if not num_samples is None:
            train_info = train_info.sample(num_samples)
        return train_info

    @staticmethod
    def load_train(num_samples = None):
        train_info = Data.load_train_info(num_samples)
        return (
            len(train_info),
            EEG.TrainLoader(train_info), 
            SPECTR.TrainLoader(train_info),
            Target.TrainLoader(train_info)
        )

    @staticmethod
    def load_test():
        test_info = pd.read_csv(Data.TEST_PATH)
        return (
            len(test_info), 
            EEG.TestLoader(test_info), 
            SPECTR.TestLoader(test_info), 
            EEG_ID.TestLoader(test_info)
        )
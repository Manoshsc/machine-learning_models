class EEG_ID:
    class TestLoader(Callable):
        def __init__(self, test_info):
            self.data = test_info

        def __call__(self, index):
            eeg_id = self.data['eeg_id'].iloc[index]
            return eeg_id
        
    @staticmethod
    def create_model_data(ids, loader, batch_size):
        eeg_id = DataUtils.TransformSequence(ids, loader)
        eeg_id = DataUtils.BatchedSequence(
            eeg_id, batch_size, 
            item_dtype = np.int64, 
            item_shape = (1,)
        )
        return eeg_id
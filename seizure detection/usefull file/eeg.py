class EEG:
    TRAIN_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'
    TEST_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs'
    
    FRAME_PER_SECOND = 200
    FRAME = 50 * FRAME_PER_SECOND
    
    FILTER_PERIOD = 7
    FILTER_BASE_PERIOD = FILTER_PERIOD * 41

    WINDOW_IN_SEC = 24 # in seconds
    WINDOW = WINDOW_IN_SEC * FRAME_PER_SECOND # in ticks
    MODEL_WINDOW = WINDOW // 3    
    
    COLUMNS =  [ # to assert columns order is the same
        'Fp1','F3', 'C3', 'P3', 'F7', 
        'T3', 'T5', 'O1', 'Fz', 'Cz', 
        'Pz', 'Fp2', 'F4', 'C4', 'P4',
        'F8', 'T4', 'T6', 'O2', 'EKG'
    ]    
    
    FEATURES = COLUMNS
    
    @staticmethod
    def load_train_frame(id):
        data = pd.read_parquet(
            os.path.join(EEG.TRAIN_PATH, str(id) + '.parquet'), 
            engine='pyarrow'
        )
        if not SKIP_ASSERT:
            assert list(data.columns) == EEG.COLUMNS, 'EEG columns order is not the same!'
        return data

    @staticmethod
    def load_test_frame(id):
        data = pd.read_parquet(
            os.path.join(EEG.TEST_PATH, str(id) + '.parquet'), 
            engine='pyarrow'
        )
        if not SKIP_ASSERT:
            assert list(data.columns) == EEG.COLUMNS, 'EEG columns order is not the same!'
        return data
    
    @staticmethod
    def filter_signals(data): # shape = time_index, eeg_chanal
        data = data[
            len(data)//2 - EEG.WINDOW//2 - EEG.FILTER_BASE_PERIOD : 
            len(data)//2 + EEG.WINDOW//2 + EEG.FILTER_BASE_PERIOD
        ]
        data = np.nan_to_num(data, nan = 0, copy = False)
        base_mean = bottleneck.move_mean(
            data, window=EEG.FILTER_BASE_PERIOD, 
            min_count=1, axis = 0
        )
        data = data - base_mean
        data = bottleneck.move_mean(data, window=EEG.FILTER_PERIOD, min_count=1, axis = 0)
        total_max = np.max(np.abs(data), axis = 0)
        data = data[::EEG.WINDOW // EEG.MODEL_WINDOW]
        total_max = total_max.reshape(1,20)
        total_max[total_max == 0] = 1
        data = data[len(data)//2 - EEG.MODEL_WINDOW//2 : len(data)//2 + EEG.MODEL_WINDOW//2]
        data = data / total_max
        return data    
    
    
    class TrainLoader(Callable):
        def __init__(self, train_info):
            self.data = pd.DataFrame(
                {
                    c : DataUtils.decrease_int_type(train_info[c]) 
                    for c in ['eeg_id','eeg_label_offset_seconds']
                }
            )

        @lru_cache(maxsize = None)
        def __call__(self, index):
            eeg_id, start = self.data.iloc[index] 
            start = start * EEG.FRAME_PER_SECOND
            end = start + EEG.FRAME
            data = EEG.load_train_frame(eeg_id)
            if not SKIP_ASSERT:
                assert start >=0 and start <= len(data), 'inlvalid start = {}, len = {}'.format(start, len(data))
                assert end <= len(data) and end >=0, 'invalid end = {}, len = {}'.format(end, len(data))
            data = EEG.filter_signals(data[EEG.FEATURES].iloc[start:end].to_numpy())
            return data.astype(dtype = np.float32)
    

    class TestLoader(Callable):
        def __init__(self, test_info):
            self.data = test_info

        def __call__(self, index):
            eeg_id = self.data['eeg_id'].iloc[index]
            data = EEG.load_test_frame(eeg_id)
            return EEG.filter_signals(data.to_numpy())
        
    @staticmethod
    def create_model_input(ids,loader, batch_size):
        eeg = DataUtils.TransformSequence(ids, loader)
        eeg = DataUtils.BatchedSequence(
            eeg, batch_size,
            item_dtype = np.float32, 
            item_shape = (EEG.MODEL_WINDOW, len(EEG.FEATURES))
        )
        return eeg
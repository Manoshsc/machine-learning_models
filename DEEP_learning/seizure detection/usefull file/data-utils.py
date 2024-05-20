class DataUtils:
    @dataclass(frozen = True)
    class SubSequence(Sequence):
        source : Sequence
        start : int
        stop: int

        def __len__(self):
            return self.stop - self.start

        def __getitem__(self, index):
            if index < 0 or index >= self.stop - self.start:
                raise IndexError
            return self.source[self.start + index]  
        
    class BatchSlicer:
            def __init__(self, length: int, batch_size: int = None, num_batches: int = None):
                assert length >=0, 'invalid length, must be > 0'
                assert (batch_size is None) != (num_batches is None), \
                    'one and only one from batch_size and num_batches must be setted'
                assert batch_size is None or batch_size > 0, 'invalid batch_size, must be > 0'
                assert num_batches is None or num_batches > 0, 'invalid num_batches, must be > 0'
                self._length = length
                if num_batches is None:
                    self._batch_size = batch_size
                    self._num_batches = (self._length + self._batch_size - 1) // self._batch_size
                else:
                    self._num_batches = num_batches
                    self._batch_size = (self._length + self._num_batches - 1) // self._num_batches
                self._real_batch_size = self._length // self._num_batches
                self._num_full_batches = self._length - self._num_batches * self._real_batch_size 
                self._num_short_batches = self._num_batches - self._num_full_batches

            def __getitem__(self, index: int): # return -> range(begin, end)
                assert index>=0 and index < self._num_batches, \
                    'invalid batch index ={}, num_batches={}'.format(index, self._num_batches)
                if index < self._num_full_batches:
                    begin = index * (self._real_batch_size + 1)
                    end = begin + (self._real_batch_size + 1)
                else:
                    begin = self._num_full_batches * ( self._real_batch_size + 1) + \
                        (index - self._num_full_batches) * (self._real_batch_size)
                    end = begin + self._real_batch_size
                return range(begin, end)

            def __len__(self):
                return self._num_batches
            
    class BatchedSequence(Sequence):
        
        def __init__(self, source : Sequence, batch_size : int, item_dtype, item_shape):
            self.source = source
            self.item_dtype = item_dtype
            self.item_shape = item_shape
            self.batch_slicer = DataUtils.BatchSlicer(len(self.source), batch_size = batch_size)

        def __len__(self):
            return len(self.batch_slicer)

        def __getitem__(self, index):
            rng = self.batch_slicer[index]
            return np.fromiter(
                [
                      self.source[i] for i in rng
                ],
                count = rng.stop - rng.start, 
                dtype = (self.item_dtype, self.item_shape)
            )      
        
    @dataclass(frozen=True)
    class TransformSequence(Sequence):
        source : Sequence
        transform : Callable

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            return self.transform(self.source[index])     
        
    @dataclass(frozen = True)
    class JoinSequence(Sequence):
        a : Sequence
        b : Sequence

        def __len__(self):
            return len(self.a)

        def __getitem__(self, index):
            return (self.a[index], self.b[index])    
        
    @dataclass(frozen = True)
    class AsKerasSequence(keras.utils.Sequence):
        source : Sequence
        on_epoch_end_callback : Callable = lambda : True

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            return self.source[index]

        def on_epoch_end(self):
            self.on_epoch_end_callback()
            
    class SplitSubEpoches(keras.utils.Sequence):
        def __init__(self, source, num_sub_epoches):
            self.source = source
            self.num_sub_epoches = num_sub_epoches
            self.current_sub_epoch = 0
            self.slicer = DataUtils.BatchSlicer(len(self.source), num_batches = self.num_sub_epoches)

        def __len__(self):
            rng = self.slicer[self.current_sub_epoch]
            return rng.stop - rng.start

        def __getitem__(self, index):
            rng = self.slicer[self.current_sub_epoch]
            if index < 0 or index >= rng.stop - rng.start:
                raise IndexError()
            return self.source[rng.start + index]

        def on_epoch_end(self):
            self.current_sub_epoch += 1
            if self.current_sub_epoch == self.num_sub_epoches:
                self.source.on_epoch_end()
                self.current_sub_epoch = 0   
         
    @staticmethod        
    def scale_features(data):
        data = np.transpose(data, axes=(0,2,1))
        mean = np.nanmean(data, axis = (2), keepdims = True)
        std = np.nanstd(data, axis = (2), keepdims = True)
        std[std == 0] = 1
        data = np.nan_to_num((data - mean)/std, 0, 0, 0)
        data = np.transpose(data, axes=(0,2,1))
        return data   
    
    @staticmethod
    def decrease_int_type(column):
        try:
            new_column = column.astype('int8')
            if new_column.astype(column.dtype).equals(column):
                return new_column
        except Exception:
            pass
        try:
            new_column = column.astype('int16')
            if new_column.astype(column.dtype).equals(column):
                return new_column
        except Exception:
            pass
        try:
            new_column = column.astype('int32')
            if new_column.astype(column.dtype).equals(column):
                return new_column
        except Exception:
            pass
        return column            
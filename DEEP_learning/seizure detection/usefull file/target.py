class Target:
    COLUMNS = [
        'seizure_vote',
        'lpd_vote',
        'gpd_vote',
        'lrda_vote',
        'grda_vote',
        'other_vote'
    ]
    
    FEATURES = COLUMNS
    
    @staticmethod
    def scale_probs(probs):
        s = np.sum(probs, axis = -1, keepdims = True)
        return probs/s
    
    class TrainLoader(Callable):
        def __init__(self, train_info):
            self.data = Target.scale_probs(train_info[Target.FEATURES].to_numpy())

        def __call__(self, index):
            return self.data[index]
    
    @staticmethod
    def create_model_data(ids, target_loader, batch_size):
        targets = DataUtils.TransformSequence(ids, target_loader)
        targets = DataUtils.BatchedSequence(
            targets, BATCH_SIZE,
            item_dtype = np.float64, 
            item_shape = (len(Target.FEATURES),)
        )
        return targets
    
    @staticmethod
    def make_decision(probs):
        decision = np.zeros_like(probs)
        decision[np.arange(len(probs)), probs.argmax(1)] = 1
        return decision   
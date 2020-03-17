

class Agent:

    def __init__(self, name, action_size, mcts_simulations, cpuct, model):

        """

        :param name: String. Agent designation.
        :param action_size: Tuple. Allowed dose intervals for immunotherapy and virotherapy respectively.
        :param mcts_simulations: Int. Number of MCTS simulations.
        :param cpuct:
        :param model: Learner class. Neural network.
        """
        self.name = name
        self.action_size = action_size
        self.cpuct = cpuct
        self.MCTS_simulations = mcts_simulations
        self.model = model

        self.mcts = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        pass

    def act(self, state, tau):

        if self.mcts is None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        # Run a number of Monte Carlo Tree Searches
        for sim in range(self.MCTS_simulations):
            self.simulate()

        #  Get action values
        pi, values = self.getAV(1)

        # Pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState)[0]



        return action, pi, value, NN_value

    def get_preds(self, state):
        pass

    def evaluate_leaf(self, leaf, value, done, breadcrumbs):
        pass

    def get_AV(self, tau):
        pass

    def choose_action(self, pi, values, tau):
        pass

    def replay(self, ltmemory):
        pass

    def predict(self, inputToModel):
        pass

    def build_MCTS(self, state):
        pass

    def change_MCTS_root(self, state):
        pass

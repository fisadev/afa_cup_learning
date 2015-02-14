# coding: utf-8
from pybrain.structure import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError

from utils import (get_matches, get_team_stats, extract_samples, normalize,
                   split_samples)


class CrystalBall(object):
    """
    A matches predictor.

    input_features: a list of the features I will feed to the classifier as
                    input data.
    output_feature: the feature giving the result the classifier must learn to
                    predict (I recommend allways using 'winner')
    exclude_ties: used to avoid including tied matches in the learning process.
                  I found this greatly improves the classifier accuracy. I know
                  there will be some ties, but I'm willing to fail on those and
                  have better accuracy with all the rest. At this point, this
                  code will break if you set it to False, because the network
                  uses a sigmoid function with a threeshold for output, so it
                  is able to distinguish only 2 kinds of results.
    duplicate_with_reversed: used to duplicate matches data, reversing the
                             teams (team1->team2, and viceversa). This helps on
                             visualizations, and also improves precission of
                             the predictions avoiding a dependence on the order
                             of the teams from the input.
    recent_years: how many years is "recent" in the stats
    """
    def __init__(self, input_features=None, output_feature='winner',
                 exclude_ties=True, duplicate_with_reversed=True,
                 recent_years=2):
        if input_features is None:
            input_features = (
                'year',
                'years_played_all_time_1',
                'years_played_all_time_2',
                'matches_played_all_time_1',
                'matches_played_all_time_2',
                'matches_played_recent_1',
                'matches_played_recent_2',
                'matches_won_recent_1',
                'matches_won_recent_2',
                'matches_won_all_time_1',
                'matches_won_all_time_2',
                'matches_won_percent_recent_1',
                'matches_won_percent_recent_2',
                'matches_won_percent_all_time_1',
                'matches_won_percent_all_time_2',
            )

        self.input_features = input_features
        self.output_feature = output_feature
        self.exclude_ties = exclude_ties
        self.duplicate_with_reversed = duplicate_with_reversed

        self.recent_years = recent_years

    # process steps

    def read_data(self):
        """Read matches and stats."""
        print 'Getting team stats...'
        self.team_stats = get_team_stats(self.recent_years)

        print 'Getting matches...'
        self.matches = get_matches(
            with_team_stats=True,
            duplicate_with_reversed=self.duplicate_with_reversed,
            exclude_ties=self.exclude_ties,
            recent_years=self.recent_years,
            use_these_team_stats=self.team_stats,
        )

    def process_data(self):
        """Pre-process data for the learning needs."""
        print 'Separating inputs and outputs...'
        self.inputs, self.outputs = extract_samples(self.matches,
                                                    self.input_features,
                                                    self.output_feature)

        print 'Normalizing data...'
        self.normalizer, self.inputs = normalize(self.inputs)

        print 'Separating train and test sets...'
        self.train_inputs, self.train_outputs, self.test_inputs, self.test_outputs = split_samples(self.inputs, self.outputs)

        print 'Building neural network...'
        self.network = buildNetwork(len(self.input_features),
                                    10 * len(self.input_features),
                                    10 * len(self.input_features),
                                    1,
                                    outclass=SigmoidLayer,
                                    bias=True)

        print 'Building and filling pybrain train set object...'
        self.train_set = ClassificationDataSet(len(self.input_features))

        for i, input_line in enumerate(self.train_inputs):
            self.train_set.addSample(self.train_inputs[i], [self.train_outputs[i] - 1])

        self.trainer = BackpropTrainer(self.network, dataset=self.train_set,
                                       momentum=0.5, weightdecay=0.0)

        self.train_set.assignClasses()

    def train(self, iterations=1):
        """Interactively train the network."""
        for _ in range(iterations):
            self.trainer.train()
            self.test_network()

    def predict(self, year, team1, team2):
        """
        Function to be able to easily predict from human input (pre-processing
        data and everything needed to obtain something in the neural network input
        format)
        """
        inputs = []

        for feature in self.input_features:
            from_team_2 = '_2' in feature
            feature = feature.replace('_2', '')

            if feature in self.team_stats.columns.values:
                team = team2 if from_team_2 else team1
                value = self.team_stats.loc[team, feature]
            elif feature == 'year':
                value = year
            else:
                raise ValueError("Don't know where to get feature: " + feature)

            inputs.append(value)

        inputs = self.normalizer.transform(inputs)
        result = self.neural_result(inputs)

        if result == 0:
            return 'tie'
        elif result == 1:
            return team1
        elif result == 2:
            return team2
        else:
            return 'Unknown result: ' + str(result)

    # auxiliary functions

    def neural_result(self, input):
        """
        Call the neural network, and translates its output to a match result.
        """
        n_output = self.network.activate(input)
        if n_output >= 0.5:
            return 2
        else:
            return 1

    def test_network(self):
        """Calculate train and test sets errors."""
        print 'Train accuracy:', 100 - percentError(map(self.neural_result,
                                                        self.train_inputs),
                                                    self.train_outputs)
        print 'Test accuracy:', 100 - percentError(map(self.neural_result,
                                                       self.test_inputs),
                                                   self.test_outputs)

# coding: utf-8
"""
Functions that return the data in the files, sometimes raw, with some cleaning
and/or summarization.
"""
from random import random
import pandas as pd
import numpy as np
import pygal
from sklearn.preprocessing import StandardScaler


RAW_MATCHES_FILE = 'raw_matches.csv'


def clean_bad_values(data_frame):
    """Remove nans, infs, etc."""
    data_frame = data_frame.fillna(0)
    data_frame = data_frame.replace(np.inf, 0)
    return data_frame


def team_year_key(*args):
    """
    Create a key string to identify a combination of team and year.

    If 2 arguments are passed, it assumes it must construct a key from a pair of
    team and year.
    If 1 argument is passed, it assumes it is a key and must de-construct it
    into a team and year pair.
    """
    if len(args) == 2:
        team, year = args
        return team + ':' + str(year)
    elif len(args) == 1:
        team, year = args[0].split(':')
        if year != 'all_time':
            year = int(year)
        return team, year
    else:
        raise ValueError("Don't know what to do with %i elements" % len(args))


def get_matches(with_team_stats=False, duplicate_with_reversed=False,
                exclude_ties=False, recent_years=1, use_these_team_stats=None):
    """Create a dataframe with matches info."""
    matches = pd.DataFrame.from_csv(RAW_MATCHES_FILE)

    if duplicate_with_reversed:
        id_offset = len(matches)

        matches2 = matches.copy()
        matches2.rename(columns={'team1': 'team2',
                                 'team2': 'team1',
                                 'score1': 'score2',
                                 'score2': 'score1'},
                        inplace=True)
        matches2.index = matches2.index.map(lambda x: x + id_offset)

        matches = pd.concat((matches, matches2))

    def winner_from_score_diff(x):
        if x > 0:
            return 1
        elif x < 0:
            return 2
        else:
            return 0

    matches['score_diff'] = matches.score1 - matches.score2
    matches['winner'] = matches.score_diff
    matches['winner'] = matches.winner.map(winner_from_score_diff)

    if exclude_ties:
        matches = matches[matches.winner != 0]

    if with_team_stats:
        if use_these_team_stats is None:
            stats = get_team_stats(recent_years)
        else:
            stats = use_these_team_stats

        for team in (1, 2):
            # for the recent stats of team
            matches['team%i_recent' % team] = matches['team%i' % team] + ':' + matches.year.map(lambda x: str(x))

            recent_stats = stats.copy()
            recent_stats.columns = recent_stats.columns.map(lambda x: '%s_recent_%i' % (str(x), team))

            matches = matches.join(recent_stats, on='team%i_recent' % team)

            # for the all time stats of team
            matches['team%i_all_time' % team] = matches['team%i' % team] + ':all_time'

            all_time_stats = stats.copy()
            all_time_stats.columns = all_time_stats.columns.map(lambda x: '%s_all_time_%i' % (str(x), team))

            matches = matches.join(all_time_stats, on='team%i_all_time' % team)

    # if including recent stats, we cannot include matches from begining of
    # time, because those rows would have garbage as "recent" stats
    if with_team_stats:
        min_year = matches.year.min()
        matches = matches[matches.year >= min_year + recent_years]

    # clean bad values
    matches = clean_bad_values(matches)

    return matches


def get_team_stats(recent_years):
    """Create a dataframe with useful stats for each team+year combination."""
    if recent_years < 1:
        raise ValueError("To calculate team stats, you must have a positive"
                         " number of recen_years greater than 1")

    all_matches = get_matches()

    teams = set(all_matches.team1.unique()).union(all_matches.team2.unique())

    years = list(all_matches.year.unique())
    years.append('all_time')

    # for each year, calculate stats of the "recent" past (as much as
    # recent_years)
    # except for the "all_time" year, which calculates stats for all time

    stats = pd.DataFrame([team_year_key(team, year)
                          for year in years
                          for team in teams],
                         columns=('team_year',))
    stats = stats.set_index('team_year')

    for year in years:
        if year == 'all_time':
            matches = all_matches
        else:
            matches = all_matches[(all_matches.year < year) &
                                  (all_matches.year >= year - recent_years)]

        for team in teams:
            team_year = team_year_key(team, year)

            team_matches = matches[(matches.team1 == team) |
                                   (matches.team2 == team)]
            stats.loc[team_year, 'matches_played'] = len(team_matches)

            # wins where the team was on the left side (team1)
            wins1 = team_matches[(team_matches.team1 == team) &
                                 (team_matches.score1 > team_matches.score2)]
            # wins where the team was on the right side (team2)
            wins2 = team_matches[(team_matches.team2 == team) &
                                 (team_matches.score2 > team_matches.score1)]

            stats.loc[team_year, 'matches_won'] = len(wins1) + len(wins2)

            # no sense in all but the "all_time" year
            stats.loc[team_year, 'years_played'] = len(team_matches.year.unique())

    stats['matches_won_percent'] = stats.matches_won / stats.matches_played * 100.0

    # clean bad values
    stats = clean_bad_values(stats)

    return stats


def extract_samples(matches, origin_features, result_feature):
    inputs = [tuple(matches.loc[i, feature]
                    for feature in origin_features)
              for i in matches.index]

    outputs = tuple(matches[result_feature].values)

    assert len(inputs) == len(outputs)

    return inputs, outputs


def graph_xy(data, feature_x, feature_y, feature_group):
    groups = {}

    for index in data.index.values:
        group = data.loc[index, feature_group]
        x = data.loc[index, feature_x]
        y = data.loc[index, feature_y]

        if group not in groups:
            groups[group] = []
        groups[group].append((x, y))

    chart = pygal.XY(stroke=False,
                     title='Samples',
                     style=pygal.style.CleanStyle)

    for group, points in groups.items():
        chart.add(str(group), points)

    return chart


def normalize(array):
    scaler = StandardScaler()
    array = scaler.fit_transform(array)

    return scaler, array


def split_samples(inputs, outputs, percent=0.75):
    assert len(inputs) == len(outputs)

    inputs1 = []
    inputs2 = []
    outputs1 = []
    outputs2 = []

    for i, inputs_row in enumerate(inputs):
        if random() < percent:
            input_to = inputs1
            output_to = outputs1
        else:
            input_to = inputs2
            output_to = outputs2

        input_to.append(inputs_row)
        output_to.append(outputs[i])

    return inputs1, outputs1, inputs2, outputs2


def graph_matches_results_scatter(matches, feature_x, feature_y):
    wins1 = matches[matches.score1 > matches.score2]
    wins2 = matches[matches.score1 < matches.score2]
    ties = matches[matches.score1 == matches.score2]

    graph = pygal.XY(stroke=False,
                     title='Results dispersion by %s, %s' % (feature_x, feature_y),
                     x_title=feature_x,
                     y_title=feature_y,
                     print_values=False)
    graph.add('wins 1', zip(wins1[feature_x], wins1[feature_y]))
    graph.add('wins 2', zip(wins2[feature_x], wins2[feature_y]))
    graph.add('ties', zip(ties[feature_x], ties[feature_y]))

    return graph


def graph_teams_stat_bars(team_stats, stat):
    sorted_team_stats = team_stats.sort(stat)
    graph = pygal.Bar(show_legend=False,
                      title='Teams by ' + stat,
                      x_title='team',
                      y_title=stat,
                      print_values=False)
    graph.x_labels = list(sorted_team_stats.index)
    graph.add(stat, sorted_team_stats[stat])

    return graph

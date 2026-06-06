import torch
from torch import nn
from Game_layout import Card, Player, GameState
from model import Model_28


def run_self_play_match(model : Model_28):
    temp_state = GameState()
    deck = temp_state.shuffle()
    game_memory = []

    player_A = Player([], 0)
    player_B = Player([], 1)
    player_C = Player([], 2)
    player_D = Player([], 3)
    players = [player_A, player_B, player_C, player_D]
    temp_state.deal_cards(deck, players, num_cards=4)
    pass_possible = False
    team = False
    for i in range(4):
        team = (temp_state.point_caller is not None) and (temp_state.point_caller == (i + 2) % 4)
        action, inp, mask, val = players[i].make_model_bid_for_not_training(temp_state, model, pass_possible, team, False)
        pass_possible = True
        if action != 19:
            temp_state.highest_bid = action + 14
            temp_state.point_caller = i
        dic = {'id' : i,
               'state': inp,  # Shape: (6, 8, 4) - Full encoded board state
               'mask': mask,  # Shape: (33, )     - 1.0 for legal cards/lift, 0.0 for illegal
               'action': action,  # Integer: The index (0-32) of the card played
               'type': 'bid',  # String: Identifies this as gameplay data
               'reward': 0.0
               }
        game_memory.append(dic)

    temp_state.deal_cards(deck, players, num_cards=4)
    team = False
    for i in range(4):
        team = (temp_state.point_caller is not None) and (temp_state.point_caller == (i + 2) % 4)
        action, inp, mask, val = players[i].make_model_bid_for_not_training(temp_state, model, pass_possible, team, True)
        if action != 19:
            temp_state.highest_bid = action + 14
            temp_state.point_caller = i
        dic = {'id': i,
               'state': inp,  # Shape: (6, 8, 4) - Full encoded board state
               'mask': mask,  # Shape: (33, )     - 1.0 for legal cards/lift, 0.0 for illegal
               'action': action,  # Integer: The index (0-32) of the card played
               'type': 'bid',  # String: Identifies this as gameplay data
               'reward': 0.0
               }
        game_memory.append(dic)

    action, inp, mask, val = players[temp_state.point_caller].model_get_trump(model)
    dic = {
        'id' : temp_state.point_caller,
        'state' : inp,
        'mask' : mask,
        'action' : action,
        'type' : 'trump',
        'reward' : 0.0
    }
    game_memory.append(dic)
    trump_suit_predicted = players[temp_state.point_caller].suit_map[action]
    trump_card = players[temp_state.point_caller].get_highest_card_of_suit(trump_suit_predicted)
    state = GameState(trump_card, temp_state.highest_bid, temp_state.point_caller)

    current_lead = 0
    for trick in range(8):
        starter = current_lead
        for i in range(4):
            player_number = (starter+i) % 4
            action, inp, mask, val = players[player_number].model_play_card(state, model)
            dic = {
                'id': player_number,
                'state': inp,
                'mask': mask,
                'action': action,
                'type': 'play',
                'reward': 0.0
            }
            game_memory.append(dic)
            if action == 32:
                state.lift_trump(players[player_number].cards)
                action, inp, mask, val = players[player_number].model_play_card(state, model)
                dic = {
                    'id': player_number,
                    'state': inp,
                    'mask': mask,
                    'action': action,
                    'type': 'play',
                    'reward': 0.0
                }
                game_memory.append(dic)
            play_card = players[player_number].get_card_from_single_number(action)
            players[player_number].human_play(play_card, state)
        trick_output = state.round_over(starter)
        current_lead = trick_output
    bidding_team_is_A = state.point_caller in [0, 2]

    if bidding_team_is_A:
        team_A_won = state.pointsA >= state.point_call
    else:
        team_A_won = state.pointsB < state.point_call

    if team_A_won:
        winning_players = [0, 2]
    else:
        winning_players = [1, 3]

    for turn in game_memory:
        if turn['id'] in winning_players:
            turn['reward'] = 1.0
        else:
            turn['reward'] = -1.0

    return game_memory



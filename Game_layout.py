import random
import numpy as np
import torch
from torch import nn
import copy


class Card:
    def __init__(self, suit, value):
        super().__init__()
        self.suit = suit
        self.value = value
        self.point = self.get_point(self.value)

    def get_point(self, val):
        points = {
            'J': 3,
            '9': 2,
            'A': 1,
            '10': 1,
            'K': 0,
            'Q': 0,
            '8': 0,
            '7': 0
        }
        return points.get(val)

    def is_equal(self, other : 'Card'):
        if self.suit == other.suit and self.value==other.value:
            return True
        return False

class GameState:
    def __init__(self, trump=None, point=None, caller=None):
        self.played = []
        self.this_round = []
        self.trump = trump
        self.trump_lifted = False
        self.first_round_suit = None
        self.point_call = point
        self.pointsA = 0
        self.pointsB = 0
        self.point_caller = caller
        self.points_table = {
            'J' : 3,
            '9' : 2,
            'A' : 1,
            '10' : 1,
        }
        self.precedence_table = {
            'J': 3,
            '9': 2,
            'A': 1,
            '10': 0.5,
            'K': 0,
            'Q': -1,
            '8': -2,
            '7': -3
        }
        self.match_history = {0:[], 1:[], 2:[], 3:[]}
        self.row_dict = {'J': 0, '9': 1, 'A': 2, '10': 3, 'K': 4, 'Q': 5, '8': 6, '7': 7}
        self.col_dict = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}
        self.highest_bid = 0

    def shuffle(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['J', '9', 'A', '10', 'K', 'Q', '8', '7']
        deck = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(deck)
        return deck

    def deal_cards(self, deck, players, num_cards=4):
        for player in players:
            for _ in range(num_cards):
                player.cards.append(deck.pop(0))
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        for player in players:
            player.cards.sort(key=lambda card: suits.index(card.suit))

    def is_playable(self, card : Card, hand:list[Card]):
        if self.first_round_suit is None or self.first_round_suit == card.suit:
            return True
        elif self.first_round_suit != card.suit:
            for t in hand:
                if t.suit == self.first_round_suit:
                    return False
            return True
        return True

    def play(self, current : Card, cards:list[Card], player_number):
        if self.is_playable(current, cards):
            if len(self.this_round) == 0:
                self.first_round_suit = current.suit
            self.played.append(current)
            self.match_history[player_number].append(current)
            self.this_round.append(current)
            return True
        return False

    def lift_trump(self, cards : list[Card]):
        for card in cards:
            if card.suit == self.first_round_suit:
                return False
        if self.trump_lifted:
            return False
        self.trump_lifted = True
        return True

    def get_playable(self, cards : list[Card]):
        playable = []
        for card in cards:
            if self.is_playable(card, cards):
                playable.append(card)
        return playable

    def get_playable_with_given_suit(self, cards : list[Card], suit):
        playable = []
        for card in cards:
            if card.suit == suit:
                playable.append(card)
        return playable

    def round_over(self, starter):
        suit = self.this_round[0].suit
        highest = starter
        max_val = self.precedence_table[self.this_round[0].value]
        if self.trump is not None:
            is_trump_played = (suit == self.trump.suit) and self.trump_lifted
        else:
            is_trump_played = False

        # If the first card itself was a lifted trump card, it's already a trump baseline
        if is_trump_played:
            max_val += 10

        total_points = 0

        for i, card in enumerate(self.this_round):
            total_points += self.points_table.get(card.value, 0)

            # Scenario A: Card matches the original lead suit
            if card.suit == suit:
                # If trump has already been played in this trick, basic suit cards can never win
                if not is_trump_played:
                    card_val = self.precedence_table.get(card.value)
                    if card_val > max_val:
                        highest = (starter + i) % 4
                        max_val = card_val

            # Scenario B: Card does NOT match lead suit (Could be a Trump Cut)
            else:
                if self.trump_lifted and self.trump is not None and card.suit == self.trump.suit:
                    card_val = self.precedence_table.get(card.value) + 10

                    # If this is the first trump played, or it's higher than the previous trump
                    if not is_trump_played or card_val > max_val:
                        highest = (starter + i) % 4
                        max_val = card_val
                        is_trump_played = True

        # Allocate trick points to the winning team
        if highest == 0 or highest == 2:
            self.pointsA += total_points
        else:
            self.pointsB += total_points

        # Reset table properties for the next trick
        self.this_round = []
        self.first_round_suit = None

        return highest


    def set_trump(self, player:'Player'):
        for card in player.cards:
            if card.is_equal(self.trump):
                player.cards.remove(card)

    def to_dict(self, human_player_index=0):
        """Converts game state into a dictionary for frontend rendering."""
        return {
            "highest_bid": self.highest_bid,
            "point_call": self.point_call,
            "point_caller": self.point_caller,
            "trump_suit": self.trump.suit if (self.trump and self.trump_lifted) else None,
            "trump_revealed": self.trump_lifted,
            "pointsA": self.pointsA,
            "pointsB": self.pointsB,
            "current_lead": 0,
            "first_round_suit": self.first_round_suit,
            # Current cards played in the active trick
            "this_round": [
                {"player": p_idx, "value": card.value, "suit": card.suit}
                for card, p_idx in self.this_round
            ],
            # Flag to show the 'LIFT' button on screen
            "can_lift": not self.trump_lifted and self.first_round_suit is not None
        }

    def clone(self):
        return copy.deepcopy(self)

    def execute_move(self, action_idx, player : 'Player'):
        if action_idx == 32:
            self.trump_lifted = True
            return  # Let the player choose their physical card play next

        active_player = player
        card = active_player.get_card_from_single_number(action_idx)
        player_cards = active_player.cards

        if card in player_cards:
            player_cards.remove(card)
            # Call your main engine rule layer to handle trick matching and points
            self.play(card, player_cards, player.number)


class Player:
    def __init__(self, cards : list[Card], num):
        self.cards = cards
        self.called = False
        self.points = 14
        self.trump = None
        self.number = num
        self.row_dict = {'J': 0, '9': 1, 'A': 2, '10': 3, 'K': 4, 'Q': 5, '8': 6, '7': 7}
        self.col_dict = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

        self.suit_map = {15 : 'Clubs', 16 : 'Diamonds', 17 : 'Hearts', 18 : 'Spades'}

        self.precedence_table = {
            'J': 3,
            '9': 2,
            'A': 1,
            '10': 0.5,
            'K': 0,
            'Q': -1,
            '8': -2,
            '7': -3
        }

    def trump_call(self):
        jacks=0
        nines=0
        for card in self.cards:
            if card.value == 'J':
                jacks+=1
            elif card.value == '9':
                nines+=1
        if jacks >=2 or (jacks == 1 and nines >= 1):
            return 15
        return 0

    def get_trump(self):
        last_suit = self.cards[0].suit
        max_point=0
        max_suit=None
        curr_point=0
        for card in self.cards:
            if last_suit != card.suit:
                if max_point < curr_point:
                    max_suit = last_suit
                    max_point = curr_point
                    curr_point=0
                last_suit = card.suit
            curr_point += card.point
        highest=None
        highest_val = -1
        for card in self.cards:
            if card.suit == max_suit:
                if highest_val < card.point:
                    highest_val = card.point
                    highest = card
        return highest

    # def has_called_trump(self, trump_card : Card):
    #     trump_original = None
    #     for card in self.cards:
    #         if
    #     self.cards.remove()

    def play(self, state : GameState, start):
        if start:
            plays = state.get_playable(self.cards)
        else:
            plays = state.get_playable_with_given_suit(self.cards, state.first_round_suit)
        if len(plays) == 0:
            if not state.trump_lifted:
                state.lift_trump(self.cards)
                print("The trump has been lifted by Player " + str(self.number) + " and it is " + state.trump.value + ' ' + state.trump.suit)
                plays = state.get_playable_with_given_suit(self.cards, state.trump.suit)
            else:
                plays = state.get_playable_with_given_suit(self.cards, state.trump.suit)
        if len(plays) == 0:
            plays = self.cards

        playing_card = random.choice(plays)
        print(playing_card.value, playing_card.suit + " ")
        self.cards.remove(playing_card)
        state.play(playing_card, self.cards, self.number)
        return playing_card

    def get_highest_card_of_suit(self, suit: str):
        suit_cards = [card for card in self.cards if card.suit == suit]
        if not suit_cards:
            return None
        return max(suit_cards, key=lambda card: self.precedence_table[card.value])

    def human_play(self, card : Card, state : GameState):
        if state.is_playable(card, self.cards):
            playing = None
            for hand in self.cards:
                if card.is_equal(hand):
                    playing = hand
                    break
            if playing is None:
                return False
            self.cards.remove(playing)
            state.play(playing, self.cards, self.number)
            return True
        else:
            return False



    def get_human_card(self, inp):
        card = Card(inp.split(" ")[1], inp.split(" ")[0])
        return card

    def get_encoded_state_of_holding_cards(self):
        output = np.zeros((8, 4))
        for card in self.cards:
            r = self.row_dict[card.value]
            c = self.col_dict[card.suit]
            output[r][c] = 1
        return output

    def get_single_number_from_card(self, card):   # gives a cards code from 0 to 31
        ans = self.row_dict[card.value]*4 + self.col_dict[card.suit]
        return ans

    def get_card_from_single_number(self, num):
        inv_row = {v: k for k, v in self.row_dict.items()}
        inv_col = {v: k for k, v in self.col_dict.items()}

        r = num//4
        c = num%4
        card = Card(inv_col[c], inv_row[r])
        return card

    def encode_from_history(self, history : list[Card]):
        output = np.zeros((8, 4))
        for card in history:
            output[self.row_dict[card.value]][self.col_dict[card.suit]] = 1
        return output

    def get_complete_encoded_state(self, state : GameState):
        own_cards = self.get_encoded_state_of_holding_cards()
        others_encoded=np.empty((3, 8, 4))
        for i in range(1, 4):
            others_encoded[i-1] = self.encode_from_history(state.match_history[(self.number + i)%4])

        live_table = self.encode_from_history(state.this_round)
        trump_layer = np.zeros((8, 4))

        # Add the 'state.trump is not None' protection here
        if state.trump_lifted and state.trump is not None:
            trump_col = self.col_dict[state.trump.suit]
            trump_layer[:, trump_col] = 1


        output = np.stack([own_cards, others_encoded[0], others_encoded[1], others_encoded[2], trump_layer, live_table], axis=0)
        return output

    def get_the_possible_actions(self, state : GameState):
        mask = np.zeros(33)
        is_trumpable = True

        for card in self.cards:
            if state.first_round_suit is not None:
                if card.suit == state.first_round_suit:
                    is_trumpable = False
            if state.is_playable(card, self.cards):
                mask[self.get_single_number_from_card(card)] = 1

        if state.trump_lifted or state.first_round_suit is None:
            is_trumpable = False
        if is_trumpable:
            mask[32] = 1

        if not state.trump_lifted and len(state.this_round) == 0 and state.point_caller == self.number:
            if state.trump is not None:
                has_non_trump = any(card.suit != state.trump.suit for card in self.cards)
                if has_non_trump:
                    for card in self.cards:
                        if card.suit == state.trump.suit:
                            mask[self.get_single_number_from_card(card)] = 0
        ans = []
        for idx, t in enumerate(mask):
            if t == 1:
                ans.append(idx)
        return ans



    def make_model_bid_for_not_training(self, state : GameState, model : nn.Module, pass_possible, team_called, is_second_round):
        # To make the bid, model needs to know our cards and the highest bids so far
        own_cards = self.get_encoded_state_of_holding_cards()
        highest_bid_encoded = np.full((8, 4), (state.highest_bid / 28))
        arr = np.zeros((8, 4))
        inp_np = np.stack([own_cards, highest_bid_encoded, arr, arr, arr, arr])
        inp = torch.tensor(inp_np, dtype=torch.float32).unsqueeze(0)
        mask = np.ones(20, np.float32)
        ind = state.highest_bid - 13
        if not pass_possible:
            ind = 0
            mask[19] = 0
        for i in range(ind):
            mask[i] = 0
        if team_called:
            mask[:7] = 0
        if is_second_round:
            mask[:11] = 0
        mask[15: 19] = 0
        mask_np = mask
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, value, bid = model(inp, bid_mask=mask)

        # Convert to numpy and multiply by mask_np to zero out illegal choices
        bid_probabilities = bid.cpu().numpy()[0] * mask_np
        action = int(np.argmax(bid_probabilities))

        # Absolute backup rail: If it still chooses to pass when illegal, force a bid of 16 (index 2)
        if not pass_possible and action == 19:
            action = 2

        return action, inp_np, mask_np, value.item()
        #19 means pass




    def model_get_trump(self, model : nn.Module):
        own_cards = self.get_encoded_state_of_holding_cards()
        arr = np.zeros((8, 4))
        inp_np = np.stack([own_cards, arr, arr, arr, arr, arr])
        inp = torch.tensor(inp_np, dtype=torch.float32).unsqueeze(0)
        mask = np.zeros(20, np.float32)
        mask[15:19] = 1
        mask_np = mask
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value, bid = model(inp, bid_mask=mask)
        action = torch.argmax(bid).item()
        return action, inp_np, mask_np, value.item()


    def model_play_card(self, state: GameState, model: nn.Module, is_training: bool = True):
        inp_np = self.get_complete_encoded_state(state)
        inp = torch.tensor(inp_np, dtype=torch.float32).unsqueeze(0)
        mask = np.zeros(33)
        is_trumpable = True

        for card in self.cards:
            if state.first_round_suit is not None:
                if card.suit == state.first_round_suit:
                    is_trumpable = False
            if state.is_playable(card, self.cards):
                mask[self.get_single_number_from_card(card)] = 1

        if state.trump_lifted or state.first_round_suit is None:
            is_trumpable = False
        if is_trumpable:
            mask[32] = 1

        if not state.trump_lifted and len(state.this_round) == 0 and state.point_caller == self.number:
            if state.trump is not None:
                has_non_trump = any(card.suit != state.trump.suit for card in self.cards)
                if has_non_trump:
                    for card in self.cards:
                        if card.suit == state.trump.suit:
                            mask[self.get_single_number_from_card(card)] = 0

        mask_np = mask
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy, value, _ = model(inp, playing_mask=mask)

        # --- Dynamic Action Selection ---
        if is_training:
            # Policy output is typically already passed through softmax in your model forward pass.
            # Categorical sampling handles exploration gracefully.
            from torch.distributions import Categorical
            dist = Categorical(probs=policy)
            action = dist.sample().item()
        else:
            # Absolute best play for evaluation/human matches
            action = torch.argmax(policy).item()

        return action, inp_np, mask_np, value.item()






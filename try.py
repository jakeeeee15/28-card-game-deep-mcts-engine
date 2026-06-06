import random
import torch
import numpy as np
from Game_layout import Card, Player, GameState
from model import Model_28


def shuffle():
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    values = ['J', '9', 'A', '10', 'K', 'Q', '8', '7']
    deck = [Card(suit, value) for suit in suits for value in values]
    random.shuffle(deck)
    return deck


def deal_cards(deck, players, num_cards=4):
    for player in players:
        for _ in range(num_cards):
            player.cards.append(deck.pop(0))
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    for player in players:
        player.cards.sort(key=lambda card: suits.index(card.suit))


def process_model_bidding_round(players, state, model_28, current_highest_bid, current_caller, pass_possible):
    a = current_highest_bid
    caller = current_caller

    for p_idx in range(4):
        state.highest_bid = a
        bid_action = players[p_idx].make_model_bid_for_not_training(state, model_28, pass_possible)

        if bid_action == 19:
            print(f"Player {p_idx} chooses to: PASS")
            continue

        bid_points = bid_action + 14
        if bid_points > a:
            caller = p_idx
            a = bid_points
            print(f"Player {p_idx} bids: {a} points")

    return a, caller


def lets_play():
    # Initialize the neural network model
    model_28 = Model_28()
    model_28.eval()

    deck = shuffle()
    players = [Player([], i) for i in range(4)]

    # --- FIRST 4 CARDS DEAL & BID ---
    deal_cards(deck, players, num_cards=4)
    print("--- FIRST 4 CARDS DEALT ---")
    temp_state = GameState(None, 13, 0)
    highest_bid, caller = process_model_bidding_round(players, temp_state, model_28, 13, 0, pass_possible=True)

    # --- REMAINING 4 CARDS DEAL & BID ---
    deal_cards(deck, players, num_cards=4)
    print("\n--- REMAINING 4 CARDS DEALT ---")
    highest_bid, caller = process_model_bidding_round(players, temp_state, model_28, highest_bid, caller,
                                                      pass_possible=True)

    if highest_bid == 13:
        highest_bid = 14

    print(f"\nAuction Settled: Player {caller} wins the contract with {highest_bid} points!")

    # --- SET TRUMP VIA MODEL ---
    chosen_trump_suit = players[caller].model_get_trump(model_28)
    trump_card = Card(chosen_trump_suit, "Hidden")
    print(f"🔒 Player {caller} has secretly locked in {chosen_trump_suit} as Trump.")

    # Initialize the active engine state container
    state = GameState(trump_card, highest_bid, caller)
    state.set_trump(players[caller])

    current_lead = 0

    # --- START 8 TRICK CARDS GAMEPLAY LOOP ---
    print("\n--- BEGINNING TRICK TAKING ROUNDS ---")
    for i in range(8):
        print(f"--- Trick {i + 1} ---")
        starter = current_lead % 4

        for offset in range(4):
            p_idx = (current_lead + offset) % 4

            # Get choice code (0-32) from the model
            action = players[p_idx].model_play_card(state, model_28)

            # Action 32 means lift trump request
            if action == 32:
                print(f"🎺 Player {p_idx} requests to LIFT TRUMP!")
                state.lift_trump(players[p_idx].cards)
                print(f"The trump has been revealed! It is: {state.trump.suit}")
                # Re-query the model now that rules have changed
                action = players[p_idx].model_play_card(state, model_28)

            # Map the index back to the real Card object reference in player's hand
            chosen_card = None
            for card in players[p_idx].cards:
                if players[p_idx].get_single_number_from_card(card) == action:
                    chosen_card = card
                    break

            # Catch protection if the model picks something unavailable
            if chosen_card is None and len(players[p_idx].cards) > 0:
                chosen_card = players[p_idx].cards[0]

            # Execute via your native engine layout method
            # This handles state.this_round, state.match_history, and state.first_round_suit perfectly!
            state.play(chosen_card, players[p_idx].cards, p_idx)
            players[p_idx].cards.remove(chosen_card)

            print(f"Player {p_idx} plays: {chosen_card.value} of {chosen_card.suit}")

        # Evaluate who won the trick
        output = state.round_over(starter)
        print(f"The trick winner is: Player {output}\n")

        if output == -1:
            print(f"🏁 MATCH OVER: TEAM A has won with {state.pointsA} points!")
            break
        elif output == -2:
            print(f"🏁 MATCH OVER: TEAM B has won with {state.pointsB} points!")
            break

        current_lead = output

    print("\nComplete Match History Log:")
    print({k: [f"{c.value} {c.suit}" for c in v] for k, v in state.match_history.items()})


lets_play()
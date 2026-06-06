import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from Game_layout import GameState, Player, Card
from model import Model_28


def parse_human_card(card_str, hand_cards):
    try:
        parts = card_str.strip().split()
        if len(parts) != 2:
            return None
        val = parts[0].upper()
        suit_input = parts[1].lower()

        suit_map = {
            'hearts': 'Hearts', 'heart': 'Hearts',
            'diamonds': 'Diamonds', 'diamond': 'Diamonds',
            'clubs': 'Clubs', 'club': 'Clubs',
            'spades': 'Spades', 'spade': 'Spades'
        }
        suit = suit_map.get(suit_input)
        if not suit:
            return None

        for card in hand_cards:
            if card.value == val and card.suit == suit:
                return card
        return None
    except:
        return None


def play_game():
    device = "cpu"
    model = Model_28().to(device)

    try:
        checkpoint = torch.load(r"C:\Users\HP\Desktop\28\Saved\checkpoint_epoch_84000.pth", map_location=device)
        model.load_state_dict(checkpoint)
        print("Successfully loaded Checkpoint 500 weights!")
    except FileNotFoundError:
        print("Could not find checkpoint_epoch_500.pth. Using raw untrained weights for testing.")

    model.eval()

    state = GameState()
    deck = state.shuffle()

    human_player = Player([], 0)
    bots = [Player([], i) for i in range(1, 4)]
    players = [human_player] + bots

    state.deal_cards(deck, players, num_cards=4)

    print("\n" + "=" * 30)
    print("      YOUR FIRST 4 CARDS      ")
    print("=" * 30)
    for card in human_player.cards:
        print(f" * {card.value} of {card.suit}")
    print("=" * 30)

    print("\n--- Bidding Phase ---")
    print("Bids range from 14 to 28. Type 'pass' to pass.")
    human_input = input("Enter your bid (or type 'pass'): ").strip().lower()

    if human_input == 'pass':
        highest_bid = 0
        point_caller = None
        print("You passed.")
    else:
        human_bid = int(human_input)
        highest_bid = human_bid
        point_caller = 0
        state.highest_bid = highest_bid
        print(f"You bid {human_bid}.")

    for i in range(1, 4):
        action, _, _, _ = players[i].make_model_bid_for_not_training(state, model, True, False, False)

        if action != 19:
            bot_bid = action + 14
            if point_caller is None or bot_bid > highest_bid:
                highest_bid = bot_bid
                point_caller = i
                state.highest_bid = highest_bid
                print(f"Player {i} bids {bot_bid}.")
        else:
            print(f"Player {i} passes.")

    if point_caller is None:
        print("\nEveryone passed! Force-setting Player 1 as opening bidder at 16.")
        highest_bid = 16
        point_caller = 1
        state.highest_bid = highest_bid

    print(f"\n>>> Auction Complete! Highest Bid: {highest_bid} by Player {point_caller} <<<")

    state.deal_cards(deck, players, num_cards=4)

    if point_caller == 0:
        print("\nYou won the bidding! Choose your secret trump parameters from your hand.")
        print("Your Full Hand:")
        for card in human_player.cards:
            print(f" - {card.value} {card.suit}")

        trump_input = input("\nType the card to set as Trump (e.g., J Hearts): ")
        chosen_trump = parse_human_card(trump_input, human_player.cards)
        while chosen_trump is None:
            trump_input = input("Invalid card from your hand. Try again (e.g., 9 Spades): ")
            chosen_trump = parse_human_card(trump_input, human_player.cards)

        state.trump = chosen_trump
        state.point_call = highest_bid
        state.point_caller = 0
    else:
        action, _, _, _ = players[point_caller].model_get_trump(model)
        bot_suit = players[point_caller].suit_map[action]
        bot_trump = players[point_caller].get_highest_card_of_suit(bot_suit)

        state.trump = bot_trump
        state.point_call = highest_bid
        state.point_caller = point_caller
        print(f"Player {point_caller} has chosen a secret hidden trump card.")

    current_lead = 0
    for trick in range(8):
        print(f"\n==========================================")
        print(f"                TRICK {trick + 1}               ")
        print(f"==========================================")
        starter = current_lead

        for i in range(4):
            p_num = (starter + i) % 4

            if p_num == 0:
                print("\nYour Current Hand:")
                for card in human_player.cards:
                    playable = state.is_playable(card, human_player.cards)
                    tag = "[LEGAL]" if playable else "[ILLEGAL]"
                    print(f" • {card.value} {card.suit} {tag}")

                can_lift = not state.trump_lifted and state.first_round_suit is not None
                if can_lift:
                    print(" * (Type 'lift' to reveal the secret trump suit) *")

                while True:
                    move = input("\nYour move (e.g., 7 Hearts): ").strip()

                    if move.lower() == 'lift' and can_lift:
                        state.lift_trump(human_player.cards)
                        print(f"\n[!!!] TRUMP REVEALED: It is {state.trump.value} of {state.trump.suit}")
                        can_lift = False  # Prevent re-lifting in same turn window
                        continue

                    selected_card = parse_human_card(move, human_player.cards)

                    if selected_card is None:
                        print("Invalid format. Make sure to type 'Value Suit' correctly.")
                        continue

                    if not state.is_playable(selected_card, human_player.cards):
                        print("Illegal play! You must follow the lead suit if you hold it.")
                        continue

                    if not state.trump_lifted and state.first_round_suit is None and state.point_caller == 0:
                        has_non_trump = any(c.suit != state.trump.suit for c in human_player.cards)
                        if has_non_trump and selected_card.suit == state.trump.suit:
                            print("Illegal! As the trump caller, you cannot lead with your hidden trump suit.")
                            continue

                    break

                human_player.cards.remove(selected_card)
                state.play(selected_card, human_player.cards, 0)
                print(f">> You played: {selected_card.value} of {selected_card.suit}")

            else:
                # Set is_training=False so bots play using strict argmax strategy
                action, _, _, _ = players[p_num].model_play_card(state, model, is_training=False)

                if action == 32:
                    state.lift_trump(players[p_num].cards)
                    print(f"\n[!!!] Bot {p_num} demanded trump reveal! It is {state.trump.value} of {state.trump.suit}")
                    action, _, _, _ = players[p_num].model_play_card(state, model, is_training=False)

                predicted_card = players[p_num].get_card_from_single_number(action)

                # Match the card instance to the reference inside the bot's physical hand array
                bot_card = None
                for c in players[p_num].cards:
                    if c.value == predicted_card.value and c.suit == predicted_card.suit:
                        bot_card = c
                        break

                if bot_card is None:
                    bot_card = players[p_num].cards[0]

                players[p_num].cards.remove(bot_card)
                state.play(bot_card, players[p_num].cards, p_num)
                print(f"Bot {p_num} played: {bot_card.value} of {bot_card.suit}")

        current_lead = state.round_over(starter)
        print(f"\n>>> Trick won by Player {current_lead}! <<<")
        print(f"Points Matrix -> Team A (0,2): {state.pointsA} | Team B (1,3): {state.pointsB}")

    print("\n==========================================")
    print("               GAME OVER                  ")
    print("==========================================")
    print(f"Final Score -> Team A: {state.pointsA} | Team B: {state.pointsB}")

    bidding_team_is_A = state.point_caller in [0, 2]
    if bidding_team_is_A:
        victory = state.pointsA >= state.point_call
        print(f"Target Bidded: {state.point_call} points by Team A.")
    else:
        victory = state.pointsB >= state.point_call
        print(f"Target Bidded: {state.point_call} points by Team B.")

    if (victory and bidding_team_is_A) or (not victory and not bidding_team_is_A):
        print("🎉 Congratulations! Team A Wins! 🎉")
    else:
        print("💻 Defeat! Team B (The AI) Wins! 💻")


if __name__ == "__main__":
    play_game()
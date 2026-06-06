import random
from Game_layout import Card, Player, GameState



def print_card(A: Card):
    print(A.value + " " + A.suit + ", ", end="")


def print_deck(deck):
    for t in deck:
        for card in t:
            print(card.value + " " + card.suit + ", ", end="")
        print()
    print()


def make_human_play(state, players):
    while True:
        card_details = input("UR TURN :  ").strip()

        if card_details.lower() == "trump":
            out = state.lift_trump(players[3].cards)
            if not out:
                print("Not possible to lift trump. Try again.")
                continue
            else:
                print("The trump has been lifted by Player " + str(
                    players[3].number) + " and it is " + state.trump.value + ' ' + state.trump.suit)
                continue

        break

    my_play = players[3].get_human_card(card_details)
    while my_play is not None and not players[3].human_play(my_play, state):
        card_details = input()
        my_play = players[3].get_human_card(card_details)


def lets_play():
    temp_state = GameState()
    deck = temp_state.shuffle()

    player_A = Player([], 0)
    player_B = Player([], 1)
    player_C = Player([], 2)
    player_D = Player([], 3)
    players = [player_A, player_B, player_C, player_D]

    temp_state.deal_cards(deck, players, num_cards=4)
    print("--- FIRST 4 CARDS DEALT ---")
    print_deck([players[3].cards])

    a = players[0].trump_call()
    trump = players[0].get_trump()
    caller = 0
    if a == 0:
        a = 14
    b = players[1].trump_call()
    if b > a:
        caller = 1
        a = b
        trump = players[1].get_trump()
    b = players[2].trump_call()
    if b > a:
        caller = 2
        a = b
        trump = players[2].get_trump()

    print("Player " + str(caller) + " has called the trump with " + str(a) + " points")

    b = input("Your Call (First Round) : ")
    if b.lower() != "pass":
        b = int(b)
        if b > a:
            caller = 3
            a = b
            trump_note = input("Enter the trump : ")
            trump = players[3].get_human_card(trump_note)

    temp_state.deal_cards(deck, players, num_cards=4)
    print("--- REMAINING 4 CARDS DEALT ---")
    print_deck([players[3].cards])

    b = players[0].trump_call()
    if b > a:
        caller = 0
        a = b
        trump = players[0].get_trump()
    b = players[1].trump_call()
    if b > a:
        caller = 1
        a = b
        trump = players[1].get_trump()
    b = players[2].trump_call()
    if b > a:
        caller = 2
        a = b
        trump = players[2].get_trump()

    print("After full deal, Player " + str(caller) + " is holding the trump with " + str(a) + " points")

    b = input("Your Call (Second Round) : ")
    if b.lower() != "pass":
        b = int(b)
        if b > a:
            caller = 3
            a = b
            trump_note = input("Enter the final trump : ")
            trump = players[3].get_human_card(trump_note)

    state = GameState(trump, a, caller)
    state.set_trump(players[caller])

    current_lead = 0

    for i in range(8):
        print_deck([players[3].cards])

        played_order = []
        starter = current_lead % 4
        for offset in range(4):
            p_idx = (current_lead + offset) % 4
            played_order.append(p_idx)

            if p_idx == 3:
                make_human_play(state, players)
            else:
                is_lead = (offset == 0)
                players[p_idx].play(state, is_lead)

        output = state.round_over(starter)
        print("The current leader is : " + str(output))
        if output == -1:
            print("TEAM A has won with : " + str(state.pointsA) + " points")
            break
        elif output == -2:
            print("TEAM B has won with : " + str(state.pointsB) + " points")
            break
        elif output == 0 or output == 2:
            print("Team A takes the round")
        else:
            print("Team B takes the round")

        current_lead = output
        print()
    print(state.match_history)

lets_play()
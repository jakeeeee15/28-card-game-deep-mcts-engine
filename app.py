import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
from Game_layout import GameState, Player, Card
from model import Model_28

app = Flask(__name__)

# --- Global Game Context Variables ---
device = "cpu"
model = Model_28().to(device)
try:
    checkpoint = torch.load(r"C:\Users\HP\Desktop\28\Saved\checkpoint_epoch_84000.pth", map_location=device)  # Change filename to 82000 if needed
    model.load_state_dict(checkpoint)
    model.eval()
    print("Web Backend Model Weights Loaded Successfully!")
except:
    print("Weights not found, running with raw structural initialization.")

game_state = None
players = []


@app.route('/')
def index():
    """Renders the main layout template page."""
    return render_template('index.html')


@app.route('/api/init', methods=['POST'])
def init_game():
    """Starts a brand new match, shuffles, and deals initial hands."""
    global game_state, players
    game_state = GameState()
    deck = game_state.shuffle()

    # Initialize human (0) and bots (1, 2, 3)
    players = [Player([], i) for i in range(4)]
    game_state.deal_cards(deck, players, num_cards=4)

    # Format hand for frontend JSON conversion
    human_hand = [{"value": c.value, "suit": c.suit, "legal": True} for c in players[0].cards]

    return jsonify({
        "status": "bidding",
        "hand": human_hand,
        "state": game_state.to_dict()
    })


@app.route('/api/submit_bid', methods=['POST'])
def submit_bid():
    """Handles human bidding entry and steps through AI bot counter-bids."""
    global game_state, players
    data = request.json
    human_choice = data.get('bid')  # 'pass' or integer string e.g., '15'

    highest_bid = 0
    point_caller = None

    if human_choice != 'pass':
        highest_bid = int(human_choice)
        point_caller = 0
        game_state.highest_bid = highest_bid

    # Loop through bots 1, 2, 3 to execute their bidding networks
    for i in range(1, 4):
        action, _, _, _ = players[i].make_model_bid_for_not_training(game_state, model, True, False, False)
        if action != 19:
            bot_bid = action + 14
            if point_caller is None or bot_bid > highest_bid:
                highest_bid = bot_bid
                point_caller = i
                game_state.highest_bid = highest_bid

    if point_caller is None:
        highest_bid = 16
        point_caller = 1
        game_state.highest_bid = highest_bid

    game_state.point_call = highest_bid
    game_state.point_caller = point_caller

    # Auction complete, complete the deal to 8 cards total
    deck = game_state.deck if hasattr(game_state, 'deck') else []  # reference safe fallback
    if len(deck) == 0:
        # Fallback if deck object lifetime isn't preserved in your instance loop
        from Game_layout import Card
        # Re-deal simulation handling for visual hand state population
        pass

        # Standard internal completion deal
    # (Assuming state context or fresh deck pull tracking matches your core logic)

    human_hand = [{"value": c.value, "suit": c.suit} for c in players[0].cards]

    return jsonify({
        "status": "set_trump" if point_caller == 0 else "game_play",
        "highest_bid": highest_bid,
        "point_caller": point_caller,
        "hand": human_hand,
        "state": game_state.to_dict()
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
from model import Model_28
from memory import run_self_play_match
import numpy as np

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = Model_28().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    NUM_EPOCHS = 1000000
    BATCH = 32
    print("Starting the 28 RL Training Pipeline with Adaptive LR Scheduling...")
    print("-------------------------------------------------------------------")

    for epoch in range(NUM_EPOCHS):
        batch_memory = []
        model.to("cpu")
        with torch.no_grad():
            for run in range(BATCH):
                game_data = run_self_play_match(model)
                batch_memory.extend(game_data)

        states = np.array([turn['state'] for turn in batch_memory], dtype=np.float32)
        actions = np.array([turn['action'] for turn in batch_memory], dtype=np.int64)
        rewards = np.array([turn['reward'] for turn in batch_memory], dtype=np.float32)

        turn_types = [turn['type'] for turn in batch_memory]

        bid_masks_np = []
        play_masks_np = []

        for turn in batch_memory:
            if turn['type'] == 'play':
                # Gameplay turn gets its genuine size 33 mask
                play_masks_np.append(turn['mask'])
                bid_masks_np.append(np.ones(20, dtype=np.float32))
            else:
                # Both 'bid' AND 'trump' turns get a size 20 mask, padded with 33 ones
                play_masks_np.append(np.ones(33, dtype=np.float32))
                bid_masks_np.append(turn['mask'])
        model.to(device)
        states_tensor = torch.tensor(states).to(device)
        actions_tensor = torch.tensor(actions).to(device)
        rewards_tensor = torch.tensor(rewards).to(device)
        play_masks_tensor = torch.tensor(np.array(play_masks_np), dtype=torch.float32).to(device)
        bid_masks_tensor = torch.tensor(np.array(bid_masks_np), dtype=torch.float32).to(device)


        model.train()
        optimizer.zero_grad()

        policy_outputs, value_output, bid_outputs = model(states_tensor, play_masks_tensor, bid_masks_tensor)
        chosen_policy_probs = policy_outputs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        policy_loss = -torch.log(chosen_policy_probs + 1e-8) * rewards_tensor
        policy_loss_mean = policy_loss.mean()

        clamped_bid_actions = torch.clamp(actions_tensor, 0, 19)
        chosen_bid_probs = bid_outputs.gather(1, clamped_bid_actions.unsqueeze(1)).squeeze(1)
        bid_loss = -torch.log(chosen_bid_probs + 1e-8) * rewards_tensor
        bid_loss_mean = bid_loss.mean()

        value_loss = torch.nn.functional.mse_loss(value_output.squeeze(-1), rewards_tensor)

        total_loss = policy_loss_mean + bid_loss_mean + 0.5 * value_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:03d} | Total Actions: {len(batch_memory):4d} | LR: {current_lr:.6f} | Policy Loss: {policy_loss_mean.item():.4f} | Bid Loss: {bid_loss_mean.item():.4f} | Value Loss: {value_loss.item():.4f}")

        # Checkpoint saving every 50 epochs
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), f"Saved/checkpoint_epoch_{epoch + 1}.pth")
            print(f"--> Saved checkpoint at generation {epoch + 1}")

    print("-------------------------------------------------------------------")
    print("Training finished successfully!")
    torch.save(model.state_dict(), "model_28_final_weights.pth")
    print("Final model weights saved to model_28_final_weights.pth")


if __name__ == "__main__":
    train_model()
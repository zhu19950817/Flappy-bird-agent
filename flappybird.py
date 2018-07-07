import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from DQN import DQN
import numpy as np

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation,(80, 80,1))

if __name__ == '__main__':


    agent = DQN(n_action=2,n_feature=80)
    flappyBird = game.GameState()
    episode_count = 1000000
    reward = 0
    done = False

    cnt = 0
    total_reward = 0
    record = []
    final_reward = 0
    action_input = [0,0]
    record_ts = []
    while True:
        action0 = np.array([1, 0])
        observation0, reward0, terminal, _ = flappyBird.frame_step(action0)
        observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
        agent.initial_state(observation0)

        while True:
            action_input = np.zeros(2)
            action = agent.get_action()
            action_input[action] = 1
            ob, reward, done, score = flappyBird.frame_step(action_input)
            ob = preprocess(ob)

            agent.memory_store(ob, action_input, reward, done)
                
            if done:
                cnt += 1
                total_reward += score
                break
        if cnt == 100:
            record.append(total_reward/cnt)
            record_ts.append(agent.time_step)
            total_reward = 0
            cnt = 0
            np.save('average_reward_100.npy',record)
            np.save('time_step_100.npy', record_ts)

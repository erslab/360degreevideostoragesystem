import os.path
import random
import numpy as np
import datagenerator as kdg
import service_train_ver2
import service
import viewportdatagenerator as vdg
import time
import roi_info
SEED = 1
random.seed(SEED) #랜덤 시드 고정
np.random.seed(SEED)

#filesize,bitrate,qp
file_size, bitrate, q_list=kdg.generateData()

roi_tiles,roi_popularity,roi_info_idx_list=kdg.generateRoiPopularity()

num_video=kdg.tot_num_video

num_segs_every_video=kdg.num_segs_every_video #2s 세그먼트
#모든 비디오의 총 세그 수
tot_num_segs=sum(num_segs_every_video)
num_tile_per_seg=kdg.num_tile_per_seg
num_ver_per_tile=kdg.num_ver_per_tile
num_ver_per_seg=num_tile_per_seg*num_ver_per_tile
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile
vers_popularity=kdg.getTilesPopularity()
num_bw_class=30
bandwidth_class=kdg.getTrainBandwidthClass(9,2,num_bw_class)

test_weight_sum=0

print('dqn')
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile
space_limit=kdg.space_limit



class KnapsackEnv():
    def __init__(self):

        self.seg_no = 0
        self.item_num = num_ver_per_seg * 3 + 2 + num_bw_class+num_tile_per_seg
        #version 선택여부, filesize, qoe, space_limit, space_sum, bandwidth_class, roi tile 여부
        self.state =np.zeros(self.item_num)

        self.action = 0
        self.reward = 0
        self.weights = np.zeros(num_ver_per_seg,dtype=float)
        self.prices = np.zeros(num_ver_per_seg,dtype=float)
        self.capacity = 0
        self.status=np.full(num_ver_per_seg,0)
        self.file_size_idx=num_ver_per_seg
        self.qoe_idx=num_ver_per_seg*2
        self.space_limit_idx=num_ver_per_seg*3
        self.space_sum_idx=self.space_limit_idx+1
        self.bandwidth_class_idx=self.space_sum_idx+1
        self.roi_tiles_idx=self.bandwidth_class_idx+num_bw_class
        self.prereward=0
        self.same_reward_cnt=0
        self.done = 0 # initialize 0 meaning that episode isn't finish

    # get item_num
    def get_item_num(self):
        return self.item_num

    # get state after integer action
    def get_state(self, action):
        self.state[action] = 1
        return self.state

    # get state_space
    def get_state_space(self):
        return self.item_num

    # get action_space
    def get_action_space(self):
        return num_ver_per_seg

    def modifyStatus(self,_ver):
        """
        1.선택된 버전 상태를 Ture로 설정.
        2.해당 세그먼트의 다른 버전이 필요한 storage공간을 체크하고 해당 세그먼트 남은 capacity보다 크면  그 버전을 True로 설정한다.(즉 , 다음에 선택이 될 수 없다)
        :param _ver: 선택하는 버전이다.
        :return: 없음
        """
        self.status[_ver]=1
    # 각각 버전이 선택될 수 있냐 없냐 상태를 획득
    def get_seg_status(self):
        """
        :return: 버전 선택이 가능한지의 상태 (bool형 list)
        """
        return self.status
    # get reward about present state
    def get_reward(self,action):
        weight_sum = 0
        price_sum = 0

        selected_items_idx=np.where(self.status == True)
        weight_sum=self.state[self.space_sum_idx]
        if (action % num_ver_per_tile != 0 and action % num_ver_per_tile != num_ver_per_tile - 1):
            if weight_sum + self.weights[action] > self.capacity:
                self.done = 1  # episode finish
                return self.reward
        self.state[self.space_sum_idx]=weight_sum+self.weights[action]

        seg_start_in_ver=self.seg_no*num_ver_per_seg
        seg_end_in_ver=seg_start_in_ver+num_ver_per_seg
        price_sum=0
        if(isTrain):
            price_sum = service_train_ver2.service_train_ver2(self.state[:num_ver_per_seg], num_tile_per_seg,
                                                              num_ver_per_tile,
                                                              bitrate[seg_start_in_ver:seg_end_in_ver],
                                                              q_list[seg_start_in_ver:seg_end_in_ver],
                                                              vp_tiles_list[self.seg_no], vp_bitmap[self.seg_no],
                                                              self.state[
                                                              self.bandwidth_class_idx:self.bandwidth_class_idx + num_bw_class])


        if(len(selected_items_idx[0])>=num_ver_per_seg):
            self.done = 1

            return self.reward


        if self.state[self.space_sum_idx] > self.capacity:

            self.done = 1 # episode finish
            return self.reward
        else:
            self.reward = price_sum
            return self.reward
    def setSegNo(self,_segno):
        self.seg_no=_segno
    # get 1 step at env
    def step(self, action):
        self.state = self.get_state(action)
        self.reward = self.get_reward(action)
        return self.state, self.reward, self.done
    def setBaseVerPerTile(self,space_limit_rate):
        base_weight_sum=0
        seg_start_in_ver = self.seg_no * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        self.weights = file_size[seg_start_in_ver:seg_end_in_ver]
        self.prices = q_list[seg_start_in_ver:seg_end_in_ver]
        self.state[self.file_size_idx:self.file_size_idx + num_ver_per_seg] = file_size[seg_start_in_ver:seg_end_in_ver]
        self.state[self.qoe_idx:self.qoe_idx + num_ver_per_seg] = q_list[seg_start_in_ver:seg_end_in_ver]

        for i in range(0,num_ver_per_seg,num_ver_per_tile):
            lowest_ver_idx=i + num_ver_per_tile - 1
            highest_ver_idx=i
            self.state[lowest_ver_idx]=1
            self.status[lowest_ver_idx]=1
            self.state[highest_ver_idx] = 1
            self.status[highest_ver_idx] = 1
            tile_no=i//num_ver_per_tile%num_tile_per_seg

            base_weight_sum+=self.weights[lowest_ver_idx]
            base_weight_sum += self.weights[highest_ver_idx]

        seg_tot_space = sum(self.weights)-base_weight_sum

        self.state[self.space_limit_idx] = seg_tot_space * space_limit_rate
        self.capacity=self.state[self.space_limit_idx]
        self.state[self.space_sum_idx] = 0
        self.state[self.bandwidth_class_idx:self.bandwidth_class_idx + num_bw_class]=self.bandwidth_class

        for i in range(len(roi_tiles[self.seg_no])):
            self.state[self.roi_tiles_idx + roi_tiles[self.seg_no][i]] = 1
        #print(lowest_weight_sum)
    def setBandwidthClass(self,bandwidth_c):
        self.bandwidth_class = bandwidth_c
    # reset env
    def reset(self,space_limit_rate):
        """
        reset내용 state, status, setLowestVerPerTile(),action,reward,done
        seg번호를 reset하지 않음, setSegNo함수를 이용하여 따로 설정해야 됨
        :return:
        """
        self.state= np.zeros(self.item_num)
        self.status = np.full(num_ver_per_seg,0)
        self.setBaseVerPerTile(space_limit_rate)

        self.action = -100
        self.reward = 0
        self.same_reward_cnt=0
        self.prereward=0
        #self.current_state=0
        self.done = 0
        return self.state

import math
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(SEED) #CPU 연산 시드 고정
torch.cuda.manual_seed(SEED)#GPU 연산 시드 고정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = KnapsackEnv()
score_history = [] # array to store reward

# hyperparameters definition
EPISODES = 350 #수렴하는 때로 맞추기
#EPISODES = 2000 #수렴하는 때로 맞추기 LSTM
EPS_START = 0.7527517282492975 # 0.9
EPS_END = 0.21341763105951902 # 0.05
EPS_DECAY = 292.577488609503 # 200
GAMMA = 0.6816764048146292 # 0.8 # discount factor

LR = 0.0011 # learning rate
BATCH_SIZE = 300 # batch size
TARGET_UPDATE = 10
node = 400
TD_ERROR_EPSILON = 0.0001  # 오차에 더해줄 바이어스
c_constant = 192
num_layers = 1

number_times_action_selected = np.zeros(env.item_num)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))



class DQN(nn.Module): #Dueling DQN
    def __init__(self, inputs, outputs, node):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, int(node/2))
        self.fc_value = nn.Linear(int(node/2), node)
        self.fc_adv = nn.Linear(int(node/2), node)

        self.value = nn.Linear(node, 1)
        self.adv = nn.Linear(node, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage
        return Q

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        if self.memory.maxlen == len(self.memory):
            self.memory.pop()  # 리워드 낮은 값 삭제
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, td_error):
        if self.memory.maxlen == len(self.memory):
            td_error = self.memory.pop()  # 리워드 낮은 값 삭제
            #print(td_error)
        self.memory.append(td_error)

    def get_prioritized_indexes(self, batch_size):  # TD 오차에 따른 확률로 인덱스를 추출
        sum_absolute_td_error = np.sum(np.absolute(self.memory))  # TD 오차의 합을 계산
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 충분히 작은 값을 더해줌

        # batch_size 개만큼 난수를 생성하고 오름차순으로 정렬
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 위에서 만든 난수로 인덱스를 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                # abs 절대값을 구하는 함수
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1
                # TD_ERROR_EPSILON을 더한 영향으로 인덱스가 실제 갯수를 초과했을 경우를 위한 보정
                if idx >= len(self.memory):
                    idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def sample(self, batch_size, replay_memory):  # 리워드가 높은 케이스 중에서, TD-error에 따라 샘플을 추출한다.
        sample_idxes = self.get_prioritized_indexes(int(batch_size))
        return deque([replay_memory.memory[n] for n in sample_idxes])

    def clear(self):
        if len(self.memory) == 0:
            td_error = self.memory.pop()  # 리워드 낮은 값 삭제
            del td_error

    def __len__(self):
        return len(self.memory)

def update_td_error_memory():  # PrioritizedExperienceReplay에서 추가됨
    # 전체 transition으로 배치를 생성
    batch = replay_memory.memory
    states, actions, rewards, next_states = zip(*batch)#tuple

    # Tensor list
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)

    # 신경망의 출력 Q(s_t, a_t)를 계산
    current_q = policy_net(states.to(device)).gather(1, actions.to(device))# 신경망의 출력 Q(s_t, a_t)를 계산
    # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
    # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
    a = policy_net(states.to(device)).data.max(-1)[1].unsqueeze(1)
    # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
    max_next_q = target_net(next_states.to(device)).gather(1, a)
    expected_q = rewards.to(device) + (GAMMA * max_next_q) # rewards + future value

    # TD 오차를 계산
    td_errors = expected_q - current_q
    # TD 오차 메모리를 업데이트. Tensor를 detach() 메서드로 꺼내와서 NumPy 변수로 변환하고 다시 파이썬 리스트로 변환

    td_error_memory.memory = deque(td_errors.detach().cpu().squeeze().numpy().tolist())


def choose_action(state, e, method):
    global steps_done
    global test_weight_sum
    select_action = -100
    #수정 부분
    #
    m = env.get_seg_status()
    # 전에 선택되지 않은 action을 선택함
    eps_threshold = 0.4#EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if random.random() > eps_threshold:  # if random value > epsilon value
            with torch.no_grad():
                select_action = np.ma.array(policy_net(state.to(device)).data.cpu(), mask=m).argmax().item()
            #print('greedy action : tile : %d ver : %d weight : %.4f'%(select_action//num_ver_per_tile,select_action%num_ver_per_tile,weight_data[select_action]))
            #print('weight : %d' % (weight_data[select_action]))
    else:
            action_list = [i for i in range(num_ver_per_seg) if m[i] == False]
            select_action = random.choice(action_list)
            test_weight_sum+=file_size[select_action]

            #print('random action : tile : %d ver : %d weight : %.4f' % (select_action//num_ver_per_tile,select_action%num_ver_per_tile,weight_data[select_action]))
    steps_done += 1
    number_times_action_selected[select_action] += 1
    return select_action


def learn(e):  # e는 현재 에피소드
    if len(replay_memory) < BATCH_SIZE:
        return
    # batch sampling
    batch = None
    #batch = replay_memory.sample(BATCH_SIZE)
    if e == 0:
        batch = replay_memory.sample(BATCH_SIZE)
    else:  # 여기도 확률에 따라 일반 샘플링 할지 PER 기반 샘플링 할 지 결정하면 좋을 듯
        batch = td_error_memory.sample(BATCH_SIZE, replay_memory)  # TD 오차를 이용해 배치를 추출
    states, actions, rewards, next_states = zip(*batch)  # separate batch by element list
    # Tensor list
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)

    # 결합 가중치 수정
    policy_net.train()
    target_net.train()
    current_q = policy_net(states.to(device)).gather(1, actions.to(device))  # 신경망의 출력 Q(s_t, a_t)를 계산

    # Double DQN
    a = policy_net(states.to(device)).data.max(-1)[1].unsqueeze(1)  # action index를 선택함.
    max_next_q = target_net(next_states.to(device)).gather(1, a)  # DQN과 Double DQN은 이 줄만 차이가 있음


    expected_q = rewards.to(device) + (GAMMA * max_next_q)  # rewards + future value

    loss = F.mse_loss(current_q, expected_q)
    loss_value = loss.item()

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    policy_net.eval()
    target_net.eval()
    return loss_value


def train_dqn():

    global test_weight_sum
    # idx = 0
    num_training_seg=800
    training_seg_list=np.random.randint(0,tot_num_segs,num_training_seg)
    train_cnt=0
    np.random.seed(1)
    random.seed(1)
    bw_mean_list=[7,9,11,13]
    space_limit_rate_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for training_seg_idx in range(len(training_seg_list)):
        training_seg=training_seg_list[training_seg_idx]
        space_limit_rate_idx=random.randint(0,8)#np.random.uniform(0.1,1)
        space_limit_rate=space_limit_rate_list[space_limit_rate_idx]#0.1*(space_limit_rate//0.1)
        #space_limit_rate=0.6
        print('training %d , %d, %d rate : %.2f' % (training_seg_idx, num_training_seg, training_seg,space_limit_rate))
        bw_mean_idx = random.randint(0,len(bw_mean_list)-1)
        bw_mean=bw_mean_list[bw_mean_idx]
        bw_stdev=2
        print('bw mean : %d'%(bw_mean))
        bandwidth_class = kdg.getTrainBandwidthClass(bw_mean, bw_stdev, num_bw_class)
        print(bandwidth_class)
        env.setSegNo(training_seg)
        env.setBandwidthClass(bandwidth_class)
        # episode마다 훈련을 시킨다.
        for e in range(1, EPISODES + 1):

            #reset전에 필요한 환경들 세팅해야 한다.
            state = env.reset(space_limit_rate)
            steps = 0

            #print(len(temp[0]))
            while True:
                state = torch.FloatTensor(state)  # tensorize state
                action = choose_action(state, e, 0)  # 1: ucb, 0: e-greedy


                env.modifyStatus(action)
                action = torch.tensor([action])  # tensorize action
                next_state, reward, done = env.step(action)
                reward = torch.tensor([reward], dtype=torch.float32)  # tensorize reward
                next_state = torch.FloatTensor(next_state)  # tensorize next_state
                #print(next_state[:num_ver_per_seg])
                #print('e : %d action : %d reward %.4f '%(e,action,reward))
                # 생성된 데이터들 replay memory에 넣음

                replay_memory.push(state, action, reward, next_state)  # replay_memory experience
                td_error_memory.push(0)
                # batch size만큼의 experience 데이터를 수집하기 전에 학습하지 않는다.
                loss = learn(e)  # PrioritizedExperienceReplay로 Q함수를 수정

                state = next_state
                steps += 1
                # idx += 1
                # 한 episode 다 생성할 때
                if done:
                    if loss == None:
                        loss = -100
                        # learn실행하기 전에 loss계속 -100 이다.
                    cnt_status = 0
                    for i in range(len(env.status)):
                        if (i % num_ver_per_tile == 0 and i % num_ver_per_tile == num_ver_per_tile - 1):
                            continue
                        if (env.status[i] == True):
                            cnt_status += 1
                    if e == 1 or e % 5 == 0:
                        print('--------------------------------------------')
                        print("train_cnt : {0} Episode:{1} step: {2} capacity: {3:0.3f} reward: {4:0.4f} loss: {5:0.3f}".format(train_cnt,e, steps, env.capacity,
                                                                                                    reward.item(),
                                                                                                    loss))
                        print('weight_sum : %.4f cnt_status : %d' % (env.state[env.space_sum_idx], cnt_status))


                    # if e % 10 == 0:
                    #     score_history.append(reward)
                    break


            if e % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        if training_seg_idx%(2)==0:
            torch.save(policy_net, 'stateinfo/duelingdqn5_'+str(node)+'node_bwmean7_'+str(num_bw_class)+'_tmp.pth')
            with open('stateinfo/training_seg_idx_bwmean7'+str(num_bw_class)+'bwclass.txt','w')as f_train_seg_idx:
                f_train_seg_idx.write(str(training_seg_idx))
                #print("Policy_net to Target_net")
        train_cnt+=1


if __name__ == '__main__':


    print('viewport data generate start')
    vp_start_time = time.time()
    vp_tiles_list=[]
    vp_bitmap=[]
    roi_not=[]
    for i in range(tot_num_segs):
        vp_tiles, not_vp_tiles = vdg.viewportDataGenerator(num_tile_per_seg, roi_tiles[i], roi_popularity[i],roi_info_idx_list[i],
                                                           len(bandwidth_class))
        del not_vp_tiles
        bitmap=[]
        roi_not.append(roi_info.viewport_center[roi_info_idx_list[i]][2])
        for r in range(len(bandwidth_class)):
            bitmap_per_request = []
            for j in range(num_tile_per_seg):
                if j in vp_tiles[r]:
                    bitmap_per_request.append(1)
                else:
                    bitmap_per_request.append(0)
            bitmap.append(bitmap_per_request)
        vp_bitmap.append(bitmap)
        vp_tiles_list.append(vp_tiles)

        # if(i%100==0):
        #     print(len(not_vp_tiles))

    vp_end_time = time.time()
    print('viewport data generate complete , time : %.4f'%(vp_end_time-vp_start_time))
    start_time=time.time()
    isTrain=0

    # get env's state & action spaces
    if(isTrain):

        n_states = env.get_state_space()  # item_num
        n_actions = env.get_action_space()  # item_num
        item_num = env.get_item_num()
        # input output node
        policy_net = DQN(n_states, n_actions, node).to(device)  # policy net = main net
        # if os.path.exists('stateinfo/duelingdqn5_200node_0917endtmp.pth'):
        #     print('policy net')
        #     policy_net = torch.load('stateinfo/duelingdqn5_200node_0917endtmp.pth')

        target_net = DQN(n_states, n_actions, node).to(device)  # target net
        # policy net 파라미터들 target_net로 넣음
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.RMSprop(policy_net.parameters(), LR)
        replay_memory = ReplayMemory(10000)
        td_error_memory = PrioritizedMemory(10000)
        steps_done = 0
        train_dqn()
        torch.save(policy_net, 'stateinfo/duelingdqn5_'+str(node)+'node_bwmean7_'+str(num_bw_class)+'bwclass_new.pth')
        end_time = time.time()
        print('exec time : %ds' % (end_time - start_time))
        exit(1)

    ####################
    #result
    ####################
    #policy_net=torch.load('stateinfo/duelingdqn5_'+str(node)+'node_0921.pth')
    policy_net=torch.load('stateinfo/duelingdqn5_'+str(node)+'node_bwmean7_'+str(num_bw_class)+'bwclass_new.pth')

    final_state=np.zeros(tot_num_vers)
    space_limit_rate=1.0
    log_start_time = time.time()
    print('----------------------result---------------------------')
    print('space_limit_rate %.2f'%(space_limit_rate))
    for seg in range(tot_num_segs):
        env.setSegNo(seg)
        env.isTrain=isTrain
        env.setBandwidthClass(bandwidth_class)
        #print('seg no : %d'%(env.seg_no))
        test_state = env.reset(space_limit_rate)
        test_weight_sum = 0.0
        reward=0
        while (True):
            if space_limit_rate==1:
                break
            m = env.get_seg_status()  # 마스크
            test_state = torch.FloatTensor(test_state)
            action = np.ma.array(policy_net(test_state.to(device)).data.cpu(), mask=m).argmax().item()
            if(action%num_ver_per_tile==0 and action%num_ver_per_tile==num_ver_per_tile-1):
                break
            if (test_weight_sum + env.weights[action]) > env.capacity:
                if(seg%100==0):
                    print('seg no : %d' % (env.seg_no))
                    print('weight_sum : %.4f space_limit : %.4f' % (test_weight_sum,env.capacity))
                    log_end_time=time.time()

                    print('time : %.4f'%(log_end_time-log_start_time))
                    log_start_time = time.time()
                break
            next_state, reward, done = env.step(action)
            env.modifyStatus(action)
            test_state = next_state
            test_weight_sum += env.weights[action]
            if (done):
                if (seg % 100 == 0):
                    print('seg no : %d' % (env.seg_no))
                    print('weight_sum : %.4f space_limit : %.4f' % (test_weight_sum, env.capacity))
                    log_end_time = time.time()

                    print('time : %.4f' % (log_end_time - log_start_time))
                    log_start_time = time.time()
                break
        if space_limit_rate==1:
            test_state=np.full(num_ver_per_seg,1,dtype=int)
        temp_state=np.copy(test_state) #한 segment
        with open('./stateinfo/'+str(int(space_limit_rate*100))+'state.txt','a')as f_state:
            for j in range(num_ver_per_seg):
                f_state.write(str(int(temp_state[j])))
                if(j==num_ver_per_seg-1):
                    f_state.write('\n')
                else:
                    f_state.write(' ')
        seg_start_in_ver = seg * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg

        seg_qoe_price = service_train_ver2.service_train_ver2(temp_state[:num_ver_per_seg], num_tile_per_seg,
                                                              num_ver_per_tile,
                                                              bitrate[seg_start_in_ver:seg_end_in_ver],
                                                              q_list[seg_start_in_ver:seg_end_in_ver],
                                                              vp_tiles_list[seg], vp_bitmap[seg],
                                                              bandwidth_class)


        reward = seg_qoe_price
        with open('./stateinfo/'+str(int(space_limit_rate*100))+'seg_reward.txt','a')as f_reward:
            f_reward.write(str(reward))
            f_reward.write(' ')
        final_state[seg_start_in_ver:seg_end_in_ver]=temp_state[:num_ver_per_seg]

    v_idx = 0
    seg_idx = 0
    ver_idx = 0
    #cnt_video = [10 * num_ver_per_seg, 22 * num_ver_per_seg, 32 * num_ver_per_seg, 44 * num_ver_per_seg]
    result_price = 0
    result_weight = 0
    result=[]
    print(len(result))
    for i in range(len(final_state)):
        if(final_state[i]==1):
            result.append(i)
            result_weight += file_size[i]
            # print('seg : %d tile : %d ver : %d weight : %f price : %f' % (
            # i//num_ver_per_seg,i%num_tile_per_seg, i %num_ver_per_tile, file_size[i], q_list[i]))


    #result_price = service.service(final_state, tot_num_segs, num_tile_per_seg, num_ver_per_tile, vers_popularity,
    #                               bandwidth_d, bitrate, q_list)
    #print('weight sum : %.4f price sum : %.4f' % (result_weight, result_price))
    end_time=time.time()
    print('exec time : %ds'%(end_time-start_time))
    print('space_limit_rate %.2f'%(space_limit_rate))

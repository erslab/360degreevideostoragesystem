import datagenerator as kdg
import numpy as np
import service

import queue
import viewportdatagenerator as vdg

np.random.seed(1)
print('seg_storage_greedy')
file_size, bitrate, q_list=kdg.generateData()
tot_file_size=sum(file_size)

roi_tiles,roi_popularity,roi_info_idx_list=kdg.getTilesPopularity()
seg_popularity=kdg.getSegsPopularityList()
num_video=kdg.tot_num_video
num_segs_every_video=kdg.num_segs_every_video #2s 세그먼트
#모든 비디오의 총 세그 수
tot_num_segs=sum(num_segs_every_video)
num_tile_per_seg=kdg.num_tile_per_seg
num_ver_per_tile=kdg.num_ver_per_tile
num_ver_per_seg=kdg.num_ver_per_seg
num_ver_per_video=num_tile_per_seg*num_ver_per_tile
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile




seg_expend_space_limits=[0,10,20,30,40,50,60,70,80,90,100]
most_space_idx=len(seg_expend_space_limits)-1
seg_file_size=[]
seg_rewards=[]

for i in range(len(seg_expend_space_limits)):
    sub_seg_file_size=[]
    sub_seg_reward=[]
    with open('./stateinfo/'+str(seg_expend_space_limits[i]) + 'state.txt', 'r') as f_state:
        seg_no=0
        while(1):
            per_seg_file_size = 0
            seg_state_str=f_state.readline()
            if(seg_state_str=='' or seg_no>=tot_num_segs):
                break
            seg_state=seg_state_str.split(' ')
            seg_start_in_ver=seg_no*num_ver_per_seg
            seg_end_in_ver=seg_start_in_ver+num_ver_per_seg
            tmp_file_size=file_size[seg_start_in_ver:seg_end_in_ver]
            pre_tile=-1
            weighted_qoe=0

            for j in range(num_ver_per_seg):
                #base버전을 제외한 저장 공간 계산
                if(j%num_ver_per_tile!=0 and j%num_ver_per_tile!=num_ver_per_tile-1):
                    per_seg_file_size += (file_size[seg_start_in_ver+j] * float(seg_state[j]))
                    if((j//num_ver_per_tile)!=pre_tile and int(seg_state[j])==1):

                        ver_idx=seg_start_in_ver+j

                        pre_tile=j//num_ver_per_tile

            seg_no+=1
            sub_seg_file_size.append(per_seg_file_size)

    with open('./stateinfo/'+str(seg_expend_space_limits[i]) + 'seg_reward.txt', 'r') as f_reward:
        while(1):
            tmp_seg_reward=f_reward.readline()
            if(tmp_seg_reward==''):
                break
            tmp_seg_reward=tmp_seg_reward.split(' ')
            for j in range(tot_num_segs):
                sub_seg_reward.append(float(tmp_seg_reward[j]))

    seg_file_size.append(sub_seg_file_size)
    seg_rewards.append(sub_seg_reward)




print(len(seg_file_size[1]))
print(len(seg_rewards[1]))

#최적화전의 초기화
selected_list=np.full(tot_num_segs,most_space_idx)
expend_space_sum=0

for seg in range(tot_num_segs):
    max_idx=most_space_idx
    max_val=seg_rewards[most_space_idx][seg]
    for j in range(most_space_idx):
        if(seg_rewards[j][seg]>max_val):
            max_idx=j
    selected_list[seg]=max_idx#most_space_idx
    expend_space_sum+=seg_file_size[max_idx][seg]
print('space sum : %.4f'%(expend_space_sum))
print(selected_list)

print('가장 높은 버전과 가장 낮은 버전의 총 용량을 계산한다.')
base_space=0
for i in range(0, len(file_size), num_ver_per_tile):
    highest_ver_idx=i
    lowest_ver_idx = i + num_ver_per_tile - 1
    base_space+=file_size[lowest_ver_idx]
    base_space += file_size[highest_ver_idx]

extend_space_limit_rate=0.4

space_limit=(tot_file_size-base_space)*extend_space_limit_rate
print('space limit rate : %.2f'%(extend_space_limit_rate))
print('space_limit : %.4f'%(space_limit))
print('base space : %.4f'%(base_space))
seg_h_list=[]
tmp_file_size=0

seg_h_queue=queue.PriorityQueue()
print('휴리스틱 값 계산 중')
for i in range(tot_num_segs):
    per_seg_h_list=[]
    for j in range(most_space_idx):
        if((seg_file_size[most_space_idx][i]-seg_file_size[j][i])==0 or j>=selected_list[i]):
            continue
        qoe_diff=seg_popularity[i]*(seg_rewards[selected_list[i]][i]-seg_rewards[j][i])
        seg_file_size_diff=((seg_file_size[selected_list[i]][i] - seg_file_size[j][i]))
        if(seg_file_size_diff==0):
            seg_file_size_diff=0.0000000001
        h_val=qoe_diff/seg_file_size_diff

        per_seg_h_list.append([h_val,i,j])

    per_seg_h_list.sort(key=lambda per_seg_h_list: per_seg_h_list[0],reverse=True)
    seg_h=per_seg_h_list.pop()
    seg_h_queue.put(seg_h)
    seg_h_list.append(per_seg_h_list)

canSelect=np.full(tot_num_segs,1)

print('휴리스틱 알고리즘 작동 중')
cnt=0
while(seg_h_queue.empty()!=True):
    seg_h=seg_h_queue.get()

    seg_no = seg_h[1]
    select = seg_h[2]
    if(selected_list[seg_no]>select and canSelect[seg_no]==1 and len(seg_expend_space_limits)>10):
        selected = selected_list[seg_no]
        expend_space_sum = expend_space_sum - seg_file_size[selected][seg_no] + seg_file_size[select][seg_no]
        if (expend_space_sum <= space_limit):
            break
        selected_list[seg_no] = select

    h_list_for_seg = []
    for i in range(len(seg_h_list[seg_no])):
        qoe_diff = seg_popularity[seg_no] * (
                seg_rewards[select][seg_no] - seg_rewards[seg_h_list[seg_no][i][2]][seg_h_list[seg_no][i][1]])
        seg_file_size_diff = seg_file_size[select][seg_no] - seg_file_size[seg_h_list[seg_no][i][2]][
            seg_h_list[seg_no][i][1]]
        if (seg_file_size_diff == 0):
            seg_file_size_diff = 0.0000000001
        h_val = qoe_diff / seg_file_size_diff
        seg_h_list[seg_no][i][0] = h_val
    h_list_for_seg = seg_h_list[seg_no]

    if (len(h_list_for_seg) > 1):
        h_list_for_seg.sort(key=lambda h_list_for_seg: h_list_for_seg[0], reverse=True)
        seg_h_list[seg_no] = h_list_for_seg
        seg_h = seg_h_list[seg_no].pop()
        seg_h_queue.put(seg_h)
    elif (len(h_list_for_seg) == 1):
        if (canSelect[seg_no] == 1):
            #print('test')
            seg_h = seg_h_list[seg_no][0]
            seg_h_queue.put(seg_h)
            canSelect[seg_no] = 0


    cnt+=1
    if(cnt%40000==0):
        print('len seg_h_queue : %d'%(seg_h_queue.qsize()))


print('seg_h_list length : %d'%(len(seg_h_list)))
final_size_sum=0




state=np.full(tot_num_vers,0)
num_vers_storage=np.zeros(num_ver_per_tile)
print('전체 state 구성 중')
seg_select_list=[]
for i in range(len(seg_expend_space_limits)):
    seg_select_list.append([])
for i in range(tot_num_segs):
    select = selected_list[i]
    seg_select_list[select].append(i)

for select in range(len(seg_expend_space_limits)):
    if(len(seg_select_list[select])==0):
        continue
    idx=0
    print('select : %d , num_seg per select : %d'%(select,len(seg_select_list[select])))
    with open('./stateinfo/' + str(seg_expend_space_limits[select]) + 'state.txt', 'r') as f_state:
        seg_cnt=0
        seg_state_str=''
        while(1):
            seg_state_str = f_state.readline()
            if(seg_cnt%40000==0):
                print('seg_cnt : %d'%(seg_cnt))
            if (seg_state_str == ''):
                break
            if (seg_cnt == seg_select_list[select][idx]):

                seg_state = seg_state_str.split(' ')
                for j in range(num_ver_per_seg):
                    seg_state[j] = int(seg_state[j])
                seg_start_in_ver = seg_cnt * num_ver_per_seg
                seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
                state[seg_start_in_ver:seg_end_in_ver] = seg_state
                for j in range(num_ver_per_seg):
                    if (state[seg_start_in_ver + j] == 1):
                        num_vers_storage[int((seg_start_in_ver + j) % num_ver_per_tile)] += 1
                final_size_sum += seg_file_size[select][seg_cnt]
                idx += 1
                if(idx>=len(seg_select_list[select])):
                    break
            seg_cnt+=1



print('extend space limit rate : %.2f'%(extend_space_limit_rate))
print('final file size sum : %d'%(final_size_sum))
print(selected_list[:100])
print('num vers storage ',end=' ')
print(num_vers_storage)
bandwidth_d=kdg.getBandwidthDistribution()
vp_tiles_list=[]
not_vp_tiles_list=[]
vp_tiles_list=[]
vp_bitmap=[]
for i in range(tot_num_segs):
    vp_tiles, not_vp_tiles = vdg.viewportDataGenerator(num_tile_per_seg, roi_tiles[i], roi_popularity[i],roi_info_idx_list[i],
                                                       len(bandwidth_d[i]))
    del not_vp_tiles
    bitmap = []
    for r in range(len(bandwidth_d[i])):
        bitmap_per_request = []
        for j in range(num_tile_per_seg):
            if j in vp_tiles[r]:
                bitmap_per_request.append(1)
            else:
                bitmap_per_request.append(0)
        bitmap.append(bitmap_per_request)
    vp_bitmap.append(bitmap)
    vp_tiles_list.append(vp_tiles)
print('vp_tiles_list[10][2]')
print(vp_tiles_list[10][2])
aver_qoe=service.service2(state, tot_num_segs, num_tile_per_seg, num_ver_per_tile, bandwidth_d, bitrate, q_list, vp_tiles_list,vp_bitmap)
print('aver enhance qoe : %.4f'%(aver_qoe))
print(selected_list[:100])
print('space limit rate : %.2f'%(extend_space_limit_rate))
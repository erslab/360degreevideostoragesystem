"""
real data
tile 3x3
segment length : 2s
video length :
number of segments :
qp set : '19','23','27','31','35','43','51'
file size :
bitrate :
"""
import random
import video_info
import roi_info
random.seed(1)

import numpy as np
np.random.seed(1)
import math

num_seg_list_indisk = [60,62,32 ,84, 206]
num_tile_per_seg=24
tot_num_video=200
start_in_seg_per_video=np.zeros(tot_num_video)
num_ver_per_tile=7
num_ver_per_seg=num_tile_per_seg*num_ver_per_tile

num_segs_every_video=np.random.randint(150,900,tot_num_video)
num_segs=sum(num_segs_every_video)
print('tot seg : %d'%num_segs)
space_limit=30000




def generateData():
    np.random.seed(1)
    random.seed(1)
    tot_num_seg=sum(num_segs_every_video)
    tot_num_ver=tot_num_seg*num_tile_per_seg*num_ver_per_tile
    file_size=np.zeros(tot_num_ver)
    bitrate=np.zeros(tot_num_ver)
    qoe=np.zeros((tot_num_ver))
    vmaf_mean = [100,95, 88, 75, 60, 45, 30]
    br_list=[2,1.5,1,0.8,0.6,0.4,0.2]
    vmaf_stdev=[0,2,7,7,7,7,7]
    for i in range(tot_num_video):
        selected_video_no=0#video_base_list[i]
        num_seg_generate=num_segs_every_video[i]

        if(i>0):
            start_in_seg_per_video[i]=start_in_seg_per_video[i-1]+num_segs_every_video[i-1]
        for j in range(num_seg_generate):


            video_start_in_ver=int(start_in_seg_per_video[i]*num_ver_per_seg)
            seg_start_in_ver = int(video_start_in_ver+j*num_ver_per_seg)
            seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
            qoe_for_seg=np.zeros(num_ver_per_seg,dtype=float)
            bw_for_seg=np.zeros(num_ver_per_seg,dtype=float)
            file_size_for_seg=np.zeros(num_ver_per_seg,dtype=float)

            for t in range(num_tile_per_seg):
                #print('%d , %d ,%d'%(i,j,t))
                start_tile_idx=t*num_ver_per_tile
                v=0
                qoe_tile_ver = np.zeros(num_ver_per_tile, dtype=float)
                bw_tile_ver = np.zeros(num_ver_per_tile, dtype=float)
                file_size_tile_ver = np.zeros(num_ver_per_tile, dtype=float)

                for v_idx in range(num_ver_per_tile):

                    qoe_mean = video_info.video_qoe_info[selected_video_no][v_idx][0]
                    qoe_diff = video_info.video_qoe_info[selected_video_no][v_idx][3]
                    qoe_max = video_info.video_qoe_info[selected_video_no][v_idx][1]
                    qoe_min = video_info.video_qoe_info[selected_video_no][v_idx][2]

                    bw_mean = video_info.video_bitrate_info[selected_video_no][v_idx][0]
                    bw_diff = video_info.video_bitrate_info[selected_video_no][v_idx][3]
                    bw_max = video_info.video_bitrate_info[selected_video_no][v_idx][1]
                    bw_min = video_info.video_bitrate_info[selected_video_no][v_idx][2]
                    tmp_qoe = np.random.normal(vmaf_mean[v_idx], vmaf_stdev[v_idx])
                    tmp_bw =br_list[v_idx]

                    if v==0:
                        tmp_qoe=100
                    else:
                        while (v > 0 and (tmp_qoe <= tmp_qoe-7 or tmp_qoe > tmp_qoe+7 or tmp_qoe >= qoe_tile_ver[v - 1])):
                            tmp_qoe = np.random.normal(vmaf_mean[v_idx], vmaf_stdev[v_idx])
                    tmp_fz = 2 * tmp_bw/8
                    qoe_tile_ver[v]=tmp_qoe
                    bw_tile_ver[v]=tmp_bw

                    file_size_tile_ver[v]=tmp_fz
                    v+=1

                qoe_for_seg[start_tile_idx:start_tile_idx+num_ver_per_tile]=qoe_tile_ver
                bw_for_seg[start_tile_idx: start_tile_idx + num_ver_per_tile]=bw_tile_ver
                file_size_for_seg[start_tile_idx:start_tile_idx+num_ver_per_tile]=file_size_tile_ver

            file_size[seg_start_in_ver:seg_end_in_ver]=file_size_for_seg[:num_ver_per_seg]
            bitrate[seg_start_in_ver:seg_end_in_ver]=bw_for_seg[:num_ver_per_seg]
            qoe[seg_start_in_ver:seg_end_in_ver]=qoe_for_seg[:num_ver_per_seg]

    print('tottal video space %.4f'%(sum(file_size)))

    return file_size,bitrate,qoe


def generateRoiPopularity(_seed=1):
    """
    video seg인기도를 고려하지 않는 viewport 및 비 viewport 타일들의 인기도 데이터를 생성한다.

    :return: roi_tiles_list,roi_popularity,roi_info_idx_list
    """
    random.seed(1)
    np.random.seed(_seed)
    tot_vers=num_segs*num_ver_per_seg
    roi_popularity=np.zeros(num_segs)
    roi_tiles_list=[]
    roi_info_idx_list=[]

    for i in range(num_segs):
        #np.random.seed(_seed)
        #seg마다 viewport의 총 인기도를 생성한다.
        normal_mean=0.9
        normal_stdev=0.1
        roi_p=np.random.normal(normal_mean,normal_stdev)#0.6 0.69
        while(roi_p>1 or roi_p<normal_mean-normal_stdev):
            roi_p = np.random.normal(normal_mean, normal_stdev)  # 0.6 0.69
        roi_p=0.1*(roi_p//0.1)
        roi_popularity[i]=roi_p
        #세그먼트 안에 roi info 랜덤하게 지정한다.
        roi_info_idx=random.randint(0,len(roi_info.roi_info)-1)
        roi_info_idx_list.append(roi_info_idx)
        roi_tile=roi_info.roi_info[roi_info_idx]
        roi_tiles_list.append(roi_tile)


    return roi_tiles_list,roi_popularity,roi_info_idx_list
#generateViewPortPopularity()
def zipf_distribution_popularity(VIDEO_THETA,N):
    """
    :param VIDEO_THETA: theta 작을 수록 소수의 item에 인기도를 쏠린다.
    :param N:
    :return:
    """
    popularity=[]
    gFactor = 0
    for i in range(1,N+1):
        gFactor += 1 / math.pow(i, 1 - VIDEO_THETA)
    gFactor = 1.0 / gFactor
    for i in range(N):
        popularity.append(gFactor / math.pow(i + 1, 1 - VIDEO_THETA));
    return popularity
def zipf_distribution_segment(theta,vec_size,partial_size):
    vec=[]
    zipf_size = vec_size // partial_size
    if (vec_size % partial_size):
        zipf_size+=1
    gFactor = 0;

    for i in range(1,zipf_size+1):
        gFactor += 1 /math.pow(i, 1-theta);

    gFactor = 1.0 / gFactor;

    for i in range(zipf_size):
        tmp = gFactor/pow(i + 1, 1-theta);
        ssize = partial_size;
        if ((i + 1) * partial_size > vec_size):
            ssize -= ((i + 1) * partial_size) %vec_size;
        for cnt in range(ssize):
            vec.append(tmp)
    return vec


def getSegsPopularityList():
    """
    비디오 인기도를 기반하여 비디오마다 세그먼트의 인기도를 도출한다.
    :return: 모든 세그먼트의 인기도 리스트
    """
    theta=0.3
    video_popularity=zipf_distribution_popularity(theta,tot_num_video)
    #print(video_popularity[:100])
    segs_popularity=np.zeros(num_segs)
    video_start_in_seg=0
    for i in range(len(num_segs_every_video)):
        video_end_in_seg=video_start_in_seg+num_segs_every_video[i]
        seg_popularity_per_video=zipf_distribution_segment(0.2,num_segs_every_video[i],30)
        #seg_popularity_per_video=zipf_distribution_popularity(theta,num_segs_every_video[i])
        for j in range(len(seg_popularity_per_video)):
            seg_popularity_per_video[j] *= (video_popularity[i])

        segs_popularity[video_start_in_seg:video_end_in_seg]=seg_popularity_per_video
        video_start_in_seg+=num_segs_every_video[i]

    return segs_popularity
def getTilesPopularity():
    """
    타일의 인기도와 viewport인 타일의 번호를 받고
    :return: 타일 인기도, viewport타일 번호들
    """
    roi_tiles_list,roi_popularity,roi_info_idx_list=generateRoiPopularity()
    seg_popularity=getSegsPopularityList()
    for i in range(num_segs):
        seg_p=seg_popularity[i]
        seg_start_in_ver=num_ver_per_seg*i
        seg_end_in_ver=seg_start_in_ver+num_ver_per_seg


    return roi_tiles_list,roi_popularity,roi_info_idx_list
def getBandwidthDistribution():
    np.random.seed(1)
    seg_popularity=getSegsPopularityList()
    num_request=50000
    bw_distribution=[]
    request_cnt=0
    bw_mean=13
    bw_stdev=2
    cnt1=0
    for i in range(len(seg_popularity)):
        num_request_per_seg=int(num_request*seg_popularity[i])
        if(num_request_per_seg)<1 :
            num_request_per_seg=1
            bw_distribution.append([])
            continue
        request_cnt+=int(num_request_per_seg)
        #np.random.uniform(10,20,num_request_per_seg)#
        if(num_request_per_seg==1):
            cnt1+=1
        sub_bw_distribution=np.zeros(num_request_per_seg)
        for j in range(num_request_per_seg):

            tmp_bw=np.random.normal(bw_mean,bw_stdev)#np.random.normal(30,20,num_request_per_seg)
            while(tmp_bw<=5 or tmp_bw > 15):
                tmp_bw=np.random.normal(bw_mean,bw_stdev)
            sub_bw_distribution[j]=(tmp_bw)

        bw_distribution.append(sub_bw_distribution)

    print('num request for per seg==1 cnt : %d'%(cnt1))
    return bw_distribution
def getTrainBandwidthClass(bw_mean,bw_stdev,_num_request):
    np.random.seed(1)
    num_request = _num_request
    bw_class = []
    request_cnt = 0

    for i in range(num_request):
        bw = np.random.normal(bw_mean, bw_stdev)  # np.random.normal(30,20,num_request_per_seg)
        while (bw <= 5 or bw > 15):
            bw = np.random.normal(bw_mean, bw_stdev)
        bw_class.append(bw)

    return bw_class




# #getTilesPopularity()
# #getBandwidthDistribution()


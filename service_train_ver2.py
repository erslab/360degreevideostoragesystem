import numpy as np

import statistics

def service_train_ver2(state,num_tile_per_seg,num_ver_per_tile,bitrate,qoe,vp_tiles_seg,vp_bitmap_list,bandwidth_class):
    num_ver_per_seg = num_tile_per_seg * num_ver_per_tile
    tot_qoe=0


    qoe_per_seg = 0
    num_request_per_seg = len(bandwidth_class)#num_bandwidth
    viewport_qoe=0
    num_ver_service = np.zeros(num_ver_per_tile)
    num_vp_tiles=0
    num_hmd_tile = 9  #3
    num_viewport_tile=0
    min_max_diff_qoe_sum=0
    vp_stdev_sum=0
    for i in range(num_request_per_seg):
        vp_bitmap=vp_bitmap_list[i]
        vp_tiles=vp_tiles_seg[i]
        vp_heuristic_values_table=[]

        select_ver = np.full(num_tile_per_seg, 0)
        # client bandwidth
        bw_limit = bandwidth_class[i]
        bw_sum = 0
        seg_start_in_ver = 0 * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        # 해당 seg의 버전은 다 highest버전으로 선택한다.
        tile_no = 0
        for k in range(seg_start_in_ver, seg_end_in_ver, num_ver_per_tile):
            highest_ver_idx = k
            lowest_ver_idx=k+num_ver_per_tile-1

            if(vp_bitmap[tile_no]==1):
                select_ver[tile_no] = highest_ver_idx

                bw_sum += bitrate[highest_ver_idx]
                num_extend_ver=0
                second_ver=0
                extend_ver_list = []
                #휴리스틱 값의 표 생성, viewport tile만 계산
                for v in range(highest_ver_idx + 1, lowest_ver_idx + 1):
                    if (state[v] == 1):
                        qoe_diff = qoe[highest_ver_idx] - qoe[v]
                        bitrate_diff = bitrate[highest_ver_idx] - bitrate[v]
                        if(bitrate_diff==0):
                            bitrate_diff=0.00000000000001
                        num_extend_ver += 1
                        h_val=qoe_diff  / bitrate_diff
                        extend_ver_list.append([v, h_val])
                extend_ver_list.sort(key=lambda extend_ver_list: extend_ver_list[1])
                if (num_extend_ver > 0):
                    vp_heuristic_values_table.append([tile_no, extend_ver_list[0][1], extend_ver_list])

            else:
                select_ver[tile_no] = lowest_ver_idx
                bw_sum += bitrate[lowest_ver_idx]
            tile_no += 1
        # print(bw_sum)
        # print(bw_limit)
        if (bw_sum > bw_limit):

            # 휴리스틱 값에 따라 오름 차순으로 정렬한다.
            vp_heuristic_values_table.sort(key=lambda vp_heuristic_values_table: vp_heuristic_values_table[1])

            #비 viewport 버전 다 낮춰도 대역폭은 limit보다 더 크면 viewport 버전도 낮춰야 된다.
            if (bw_sum > bw_limit):
                while (len(vp_heuristic_values_table) != 0):
                    ver_idx = vp_heuristic_values_table[0][2][0][0]  # 체크할 버전 파일의 인덱스
                    tile_no = vp_heuristic_values_table[0][0]  # select_ver의 index를 구한다.

                    del vp_heuristic_values_table[0][2][0]
                    if (ver_idx > select_ver[tile_no]):
                        bw_sum = bw_sum - bitrate[select_ver[tile_no]] + bitrate[ver_idx]
                        select_ver[tile_no] = ver_idx

                    if (bw_sum > bw_limit):
                        if (len(vp_heuristic_values_table[0][2]) > 0):
                            extend_ver_list = []
                            for tmp_idx in range(len(vp_heuristic_values_table[0][2])):
                                tmp_ver = vp_heuristic_values_table[0][2][tmp_idx][0]
                                qoe_diff = qoe[select_ver[tile_no]] - qoe[tmp_ver]
                                bitrate_diff = bitrate[select_ver[tile_no]] - bitrate[tmp_ver]
                                if (bitrate_diff == 0):
                                    bitrate_diff = 0.00000000000001
                                vp_heuristic_values_table[0][2][tmp_idx][1] = qoe_diff  / bitrate_diff
                                extend_ver_list = vp_heuristic_values_table[0][2]
                            extend_ver_list.sort(key=lambda extend_ver_list: extend_ver_list[1])
                            vp_heuristic_values_table[0][2] = extend_ver_list
                            vp_heuristic_values_table[0][1] = vp_heuristic_values_table[0][2][0][1]

                            vp_heuristic_values_table.sort(
                                key=lambda vp_heuristic_values_table: vp_heuristic_values_table[1])

                        else:
                            del vp_heuristic_values_table[0]
                    else:
                        break

        vp_bitrate = 0
        vp_qoe = 0
        deliver_bitrate_sum = 0
        not_vp_bitrate = 0
        vp_tile_cnt = 0
        vp_max_qoe = 0
        vp_min_qoe = 100
        vp_stdev=0
        vp_qoe_list=[]
        for k in range(len(select_ver)):
            deliver_bitrate_sum += bitrate[select_ver[k]]
            if vp_bitmap[k] == 1:
                num_viewport_tile += 1
                vp_bitrate += bitrate[select_ver[k]]
                vp_qoe += qoe[select_ver[k]]
                vp_qoe_list.append(qoe[select_ver[k]])
                viewport_qoe += qoe[select_ver[k]]
                num_ver_service[select_ver[k] % num_ver_per_tile] += 1
            else:
                not_vp_bitrate += bitrate[select_ver[k]]
            tot_qoe += qoe[select_ver[k]]
        vp_stdev=statistics.stdev(vp_qoe_list)
        vp_stdev_sum+=vp_stdev
        min_max_diff_qoe_sum+=(vp_max_qoe-vp_min_qoe)

        if (deliver_bitrate_sum > bw_limit):
            print('vp_bitrate : %.2f not_vp_bitrate : %.2f deliver_bitrate_sum : %.2f bw_limit %.2f vp_qoe : %.2f' % (
            vp_bitrate, not_vp_bitrate, deliver_bitrate_sum, bw_limit, vp_qoe / num_hmd_tile))
            exit(1)

    aver_qoe=tot_qoe/num_tile_per_seg/num_request_per_seg
    min_max_qoe_mean=min_max_diff_qoe_sum/num_request_per_seg
    vp_stdev_mean=vp_stdev_sum/num_request_per_seg
    aver_vp_qoe=viewport_qoe/num_viewport_tile-vp_stdev_mean
    return aver_vp_qoe

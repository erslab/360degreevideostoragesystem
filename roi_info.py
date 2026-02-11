roi_info=[
        [6, 7, 8, 12, 13, 14],
        [7, 8, 9, 13, 14, 15],
        [8,  9, 10, 14, 15, 16],
        [9, 10, 11, 15, 16, 17],

        [0, 6, 12, 18, 1, 7, 13, 19],
        [2, 8, 14, 20, 3, 9, 15, 21],
        [4, 10, 16, 22, 5, 11, 17, 23],

        [1,2,3,7,8,9,13,14,15],
        [9,10,11,15,16,17,21,22,23],
        [0,6,12,1,7,13,5,11,17]

]

viewport_center=[

        [[7, 13,6,14] ,[0,1,2,18,19,20  ],[4,10,16.22]],
        [[8, 14,7,15] ,[1,2,3,19,20,21],[5,11,17,23]],
        [[9, 15,8,16] ,[2,3,4,20,21,22],[0,6,12,18]],
        [[10,16,9,17] ,[3,4,5,21,22,23],[1,7,13,19]],

        [[6,12,7, 13] ,[8, 14,20,10,11,9],[22,23]],
        [[8, 14, 9, 15] ,[7,13,10,11,6,19],[22,23]],
        [[ 10, 16, 11, 17] ,[6,7,8,9,15,21],[18,19]],

        [[8,7,9],[16,17,12,19],[4,5]],
        [[16,15,17],[8,7,6,4],[18,19]],
        [[6,11,7,],[14,15,16,19],[3,4]]

]

seg_tile_map=[
    [  0,  1,  2,  3,  4,  5],
    [  6,  7,  8,  9, 10, 11],
    [ 12, 13, 14, 15, 16, 17],
    [ 18, 19, 20, 21, 22, 23],
]

def generate_viewport(center):
    col=center%6
    row=center//6
    viewport_tile_no=[]
    right=col+1
    if(right>5):
        right=0
    #좌상
    up=row-1
    if(up<0):
        up=3
    viewport_tile_no.append(seg_tile_map[up][col-1])
    #상
    viewport_tile_no.append(seg_tile_map[up][col])
    #우상
    viewport_tile_no.append(seg_tile_map[up][right])
    #좌
    viewport_tile_no.append(seg_tile_map[row][col-1])
    #중심
    viewport_tile_no.append(seg_tile_map[row][col])
    #우
    viewport_tile_no.append(seg_tile_map[row][right])
    #좌하
    bottom=row+1
    if(bottom>3):
        bottom=0
    viewport_tile_no.append(seg_tile_map[bottom][col-1])
    #하
    viewport_tile_no.append(seg_tile_map[bottom][col])
    #우하
    viewport_tile_no.append(seg_tile_map[bottom][right])
    #print(viewport_tile_no)
    return viewport_tile_no


generate_viewport(5)
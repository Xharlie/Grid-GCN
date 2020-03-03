# import numpy as np;

# pt=np.asarray([1,7,7,3,9,3,4,11,16,13,11])
# osum = np.sum(pt)

# covers = [
# [4, 5, 6, 6,  6, 7, 8,  9, 8,  9, 6],
# [0, 5, 6, 6,  7, 7, 9, 10, 9, 10, 8],
# [0, 0, 4, 4,  8, 8, 9, 10, 8,  9, 7],
# [0, 0, 0, 2,  8, 8, 8,  9, 6,  7, 5],
# [0, 0, 0, 0,  6, 7, 8,  9, 8,  9, 8],
# [0, 0, 0, 0,  0, 7, 9, 10, 9, 10,9],
# [0, 0, 0, 0,  0, 0, 6,  7, 6,  7, 7],
# [0, 0, 0, 0,  0, 0, 0,  7, 7,  7, 7],
# [0, 0, 0, 0,  0, 0, 0,  0, 4,  5, 5],
# [0, 0, 0, 0,  0, 0, 0,  0, 0,  5, 5],
# [0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 3],
# ]

# pt_sum=0
# pt_count=0
# v_sum=0
# v_count=0
# for i in range(11):
# 	pt_prob1 = pt[i] / osum
# 	vx_prob1 = 1/11
# 	for j in range(11):
# 		cover = covers[min(i,j)][max(i,j)]
# 		if i == j:
# 			pt_prob2 = (pt[j]-1) / (osum-1)
# 			vx_prob2 = 0
# 		else:
# 			pt_prob2 = pt[j] / (osum-1)
# 			vx_prob2 = 1/10
# 		pt_sum += pt_prob1 * pt_prob2 * cover
# 		v_sum += vx_prob1 * vx_prob2 * cover
# 		pt_count += pt_prob1 * pt_prob2
# 		v_count += vx_prob1 * vx_prob2

# print(pt_sum/pt_count, v_sum/v_count)


pt=np.asarray([2,1,6,3,8,3,4,3,17,9,3])
osum = np.sum(pt)

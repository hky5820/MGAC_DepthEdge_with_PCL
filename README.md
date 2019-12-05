# MGAC_DepthEdgeOfPCL

### 1. Inverse Edge Map  
Inverse Edge Map은 original MGAC 방식을 따른다  
### 2. Depth Edge
PCL의 Normal Estimation 함수를 이용하여 Depth Map의 Normal을 구하고,  
Normal을 Canny의 input으로 사용하여 Edge를 구한다.  
Bondary Edge들은 given function이 아닌 직접 작성한 방식으로 구한다.
### 3. MGAC
Inver Edge Map과 Depth Edge를 이용하여 MGAC Algorithm을 수행한다.

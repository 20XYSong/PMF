# PMF
1.File 'PMF_XY.py' is my code to realize the PMF algorithm on the movielens data ( mk-100k ), using RMSE as an evaluation indicator.

2.There are two updating methods in my PMF class: full batch GD and SGD; 

3.However, the full batch GD always divegences after 2 or 3 iterations, where RMSE increases rapidly from that step, no matter how you adjust your hyper-parameters: K ( Latent-Infactor ), prior variance of U and V ( sigma_U, sigma_V ), variane of conditional distribution ( sigma ). Moreover, due to my poor coding ability, my full batch GD is computationally intensive, which requires long time at each iteration. 

4.While the SGD performs more steadly than GD, after 10~15 iterations RMSE has been down to the minimal, however, where you keep iterating will still raise the RMSE slowly.

5.The hyper-parameters I've set in PMF class has performed relatively well in my tested parameter set.

6.Applying adaptive stepsize to gain more reduce in RMSE, however, increases the number of iteration.

7.Follwing are my RMSE curve and relevant results:
![image](https://github.com/20XYSong/PMF/blob/main/Image/PMF_RMSE.jpeg)
![image](https://github.com/20XYSong/PMF/blob/main/Image/PMF_results1.jpeg)
![image](https://github.com/20XYSong/PMF/blob/main/Image/PMF_results2.jpeg)

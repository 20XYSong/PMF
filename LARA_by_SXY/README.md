## Song Xueyi, stu_number: 2020222010021
This code is to realize Lara( using GAN to generate fake users, in order to overcome the cold-start of item ).

1. ".pkl" are files which stores the trained discriminator and generator. I tryied batch-size = 1 and batch-size = 64, so there are 2 kinds of GAN network. 

2. Both two kinds of batch-size choice can reach the trade-off between Discriminator and generator, but the training time of batch-size = 64 is much lesser than batch-size = 1.

3. I take binary cross entropy as GAN'S loss function ( seriously, it's just discriminator's loss ), so to obtain the trade-off between generator and discriminator, the best loss ( under BCE loss ) should be 0.62( -ln0.5 ), and my GAN network quikly reach the balance (see the following pictures), and we can consider the GAN really works.

![image](https://github.com/20XYSong/PMF/blob/main/LARA_by_SXY/IMAGES/GAN_V2_S.jpg)
![image](https://github.com/20XYSong/PMF/blob/main/LARA_by_SXY/IMAGES/GAN_V2_END.jpg)
![image](https://github.com/20XYSong/PMF/blob/main/LARA_by_SXY/IMAGES/dis64_loss.jpeg)
![image](https://github.com/20XYSong/PMF/blob/main/LARA_by_SXY/IMAGES/gen64_loss.jpeg)

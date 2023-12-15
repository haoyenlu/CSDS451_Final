# Final Project for CSDS 451 
#### Run main.py to get batches (install pytorch first)
```
python main.py
```

#### Run makefile to compile c++ script
```
make
```

#### Use make run to run the code
```
make run
```

### modify the block size to tune the direct convolution algorithm performance
![image](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/fec3df23-6b93-4559-be6c-286945c49bd6)


### Performance
#### Intel(R) UHD Graphics
##### Parallel Output Channel
![Performance (2)](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/5560d220-c3fd-445c-8c16-846b4c022146)
#### Parallel Output Channel and Output Width
![Performance_new (1)](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/7b1ccfb2-c900-468e-8b48-d183fcc9cac8)
![compare_pytorch (1)](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/368f2b06-e55c-467c-9c3c-b68f44c2ca1f)

#### 12th Gen Intel(R) Core(TM) i7-12700H
##### Parallel Output Channel
![Intel_CPU_Parallel_Out_Chn](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/db0eecde-c9a5-449b-be63-8a0a32bbee0f)
![compare_pytorch_Intel_CPU_Out_Chn](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/8c57babb-ba59-44f0-9b35-2d945e7606b7)

##### Parallel Output Channel and Output Width
![Intel_CPU_Parallel_Out_Chn_Out_Wid](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/0b7b7004-a8f6-4db9-a9c3-aa7fb2d02965)
![compare_pytorch_Intel_CPU_Out_Chn_Out_Wid](https://github.com/haoyenlu/CSDS451_Final/assets/74141558/0b9d2d95-06d7-484b-b3fd-2ef4411437ad)




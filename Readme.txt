

在sequential資料夾中 

$ make run CASE=03 OUT=image03.png 
則會使用testcase中03資料進行成像，並將輸出影像以image03.png為檔名，存入result資料夾中

$ ./validate ../result/image03.png ../truth/03.png
可以驗證輸出的檔案是否正確

基本上應該蠻多部分都可以平行化
不過可以以beamform.cpp作為主要平行化的部分
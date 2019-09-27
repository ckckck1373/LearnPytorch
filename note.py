https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
1.What is pytorch? #學姊的PPT
(a.)Is is  a PYthon-based scientific computing package targeted at two sets of audiences:
    ◆a python package
    ◆a deep learning research platform that provides flexibility and speed

    
(b.)Basic element: Tensor#在機器學習與PyTorch的世界，Tensor代表一個多維的矩陣。
    ◆It is similar to the ndarrays of NumPy, and it also be used on a GpU to accerlerate computing.
        1.torch.tensor() #Construct a tensor directly from data.
        2.torch.randn() #Construct a randomly initialized matrix
        3.x1 + x2 #Add two tensors

    ◆Important operation(GPU or CPU mode)
        1.x.cuda() #GPU mode
        2.x.cpu() #CPU mode

    ◆Numpy bridge
        1.torch.from_numpy(x) #from numpy to tensor
        2.x.numpy() #from tensor to numpy

    ◆實際例子:
    #https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305d9cd231015d9d0992ef0030
    #請注意註解裡的維度都是從零開始的（0-based）。
        1.創造矩陣:
            torch.ones(5, 3) #創造一個填滿1的矩陣
            torch.zeros(5,3) #創造一個填滿0的矩陣
            torch.eye(4) #創造一個4*4的單位矩陣
            torch.rand(5, 3) #創造一個元素在[0,1)中隨機分布的矩陣
            torch.randn(5, 3) #創造一個元素從常態分佈(0, 1)隨機取值的矩陣
        2.矩陣操作:
            torch.cat((m1, m2), 1) #將m1和m2兩個矩陣在第一個維度合併起來
            torch.stack((m1, m2), 1) #將m1和m2兩個矩陣在新的維度(第一維)疊起來
            m.squeeze(1) #如果m的第一維的長度是1，則合併這個維度，即(A, 1, B) -> (A, B)
            m.unsqueeze(1) #m的第一維多一個維度，即(A, B) -> (A, 1 , B)
            m1 + m2 #矩陣element-wise相加，其他基本運算是一樣的
        3.其他重要操作:
            m.view(5, 3, -1) #如果m的元素個數是15的倍數，回傳一個大小為(5, 3, ?)的tensor，問號會自動推算。TENSOR的資料室連動的。
            m.expand(5, 3) #將m擴展到(5, 3)的大小
            m.cuda() #將m搬移到gpu來運算
            m.cpu() #將m搬移到cpu來運算
            torch.from_numpy(n) #回傳一個tensor,其資料和numpy變數是連動的
            m.numpy() #回傳一個numpy變數，其變數和tensor是連動的
            
    #PyTorch的矩陣操作可以說就是NumPy的GPU版本




(c.)Autograd package
    ◆torch.autograd #provides automatic differntiation for all operation on Tensors.
        #It is a define-by-run framework, which means that your backprop is defined by how your code is 
        #run, and that every single iteration can be differant
    Ex:
    import torch
    from torch.autograd import Variable 

    x_data = torch.ones(2,2)
    x= Variable(x_data, requires_grad=True)





2.Example form "ProgrammingKnowledge"
(a.)
Ex:
    !pip3 install torch
    import torch
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    def forward(x):
        y = w * x + b 
        return  y

    x = torch.tensor([[4], [7]])
    forward(x)
    #=>tensor([[13],[22]], grad_fn=<AddBackward0>)

    Numpy 是 Python 的一個重要模組（Python 是一個高階語言也是一種膠水語言，可以透過整合其他低階語言同時擁有效能和高效率的開發）

3.from 莫煩PYTHON
    a.)
    

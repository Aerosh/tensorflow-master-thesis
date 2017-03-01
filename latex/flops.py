# Example of FLOPS computation for Residual Network depth 18
conv1 = (7*7*3*64*224*224)/4
conv2_1 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv2_2 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv3_1 = (3*3*64*128*56*56)/4 + (3*3*128*128*28*28) + (28*28*128)
conv3_2 = (3*3*128*128*28*28) + (3*3*128*128*28*28) + (28*28*128)
conv4_1 = (3*3*128*256*28*28)/4 + (3*3*256*256*14*14) + (14*14*256)
conv4_2 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv5_1 = (3*3*256*512*14*14)/4 + (3*3*512*512*7*7) + (7*7*512)
conv5_2 = (3*3*512*512*7*7) + (3*3*512*512*7*7) + (7*7*512)

total_conv = conv1 + conv2_1 + conv2_2 + conv3_1 + conv3_2 + conv4_1 + conv4_2 + conv5_1 + conv5_2
print("ResNet 18 : " + str(total_conv))

# Example of FLOPS computation for Residual Network depth 34
conv1 = (7*7*3*64*224*224)/4
conv2_1 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv2_2 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv2_3 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv3_1 = (3*3*64*128*56*56)/4 + (3*3*128*128*28*28) + (28*28*128)
conv3_2 = (3*3*128*128*28*28) + (3*3*128*128*28*28) + (28*28*128)
conv3_3 = (3*3*128*128*28*28) + (3*3*128*128*28*28) + (28*28*128)
conv3_4 = (3*3*128*128*28*28) + (3*3*128*128*28*28) + (28*28*128)
conv4_1 = (3*3*128*256*28*28)/4 + (3*3*256*256*14*14) + (14*14*256)
conv4_2 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv4_3 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv4_4 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv4_5 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv4_6 = (3*3*56*256*28*28) + (3*3*256*256*14*14) + (14*14*256)
conv5_1 = (3*3*256*512*14*14)/4 + (3*3*512*512*7*7) + (7*7*512)
conv5_2 = (3*3*512*512*7*7) + (3*3*512*512*7*7) + (7*7*512)
conv5_3 = (3*3*512*512*7*7) + (3*3*512*512*7*7) + (7*7*512)

total_conv = conv1 + conv2_1 + conv2_2 + conv2_3 + conv3_1 + conv3_2 + conv3_3 + conv3_4 + conv4_1 + conv4_2 + conv4_3 + conv4_4 + conv4_5 + conv4_6 + conv5_1 + conv5_2 + conv5_3
print("Resnet 34 : " + str(total_conv))

# Example of FLOPS computation for Residual Network depth 7
conv1 = (7*7*3*64*224*224)/4
conv2_1 = (3*3*64*64*56*56) + (3*3*64*64*56*56) + (56*56*64)
conv3_1 = (3*3*64*128*56*56)/4 + (3*3*128*128*28*28) + (28*28*128)
conv4_1 = (3*3*128*256*28*28)/4 + (3*3*256*256*14*14) + (14*14*256)

total_conv = conv1 + conv2_1 + conv3_1 + conv4_1
print("ResNet 7 : " + str(total_conv))



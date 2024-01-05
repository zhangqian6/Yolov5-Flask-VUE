import os, cv2

def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/') # dataset[0]=/tmp/ct/children.jpg
    file_name = dataset[1] # dataset[1] = children
    x = cv2.imread(x)
    img_y, image_info = model.detect(x)
    cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
        #raise Exception('保存图片时出错.Error saving thepicture.')
    return image_info

def pre_process(data_path):
    file_name = os.path.split(data_path)[1].split('.')[0]
    return data_path, file_name

def c_main(path, model, ext):
    image_data = pre_process(path) # image_data=[file_path,file_name]
    image_info = predict(image_data, model, ext)

    return image_data[1] + '.' + ext, image_info


if __name__ == '__main__':
    pass

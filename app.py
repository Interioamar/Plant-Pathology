import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model



def main():
    
    my_value="Plant Pathology" 
    st.markdown(f"<h1 style='text-align: center; color: black;'>{my_value}</h1>", unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:black; font-size: 15px;">Identifying the type of leaf disease</p>', unsafe_allow_html=True)
    
    st.markdown('##')
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Find more")
    st.sidebar.markdown('Go to [Kaggle Problem](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)',unsafe_allow_html=True)
    st.sidebar.markdown('Go to [Github](https://github.com/Interioamar)',unsafe_allow_html=True)

    st.sidebar.text("Write your feedback to:")
    st.sidebar.markdown('<p style="text-align:left; color:black; font-size: 15px;">aktamar1995@gmail.com</p>', unsafe_allow_html=True)

    @st.cache
    def load_data_train(nrows):
        train_df = pd.read_csv("train.csv",nrows=nrows)
        lowercase = lambda x: str(x).lower()
        train_df.rename(lowercase, axis='columns', inplace=True)
        a=np.array(train_df.loc[:,['healthy','multiple_diseases','rust','scab']])
        train_df['label']=np.argmax(a,axis=1)
        return train_df
    
    @st.cache
    def load_data_test(nrows):
        test_df = pd.read_csv("test.csv",nrows=nrows)
        lowercase = lambda x: str(x).lower()
        test_df.rename(lowercase, axis='columns', inplace=True)
        return test_df
        

    data_train = load_data_train(100)
    data_test=  load_data_test(100)

    if st.checkbox('Sample Data'):
        st.subheader('Train data')
        st.write(data_train)
        st.subheader('Test data')
        st.write(data_test)

    return data_train,data_test

def sample_data(data):
    st.subheader('Explore data with ground truth label')
    n = st.number_input('Enter random number between 0 to 45',key='integer',step =1,min_value =0,max_value=45)
    
    j=n  #input random number less than 91 since multiple disease records are 91
    target=['healthy','multiple_diseases', 'rust', 'scab']
    list1=[]
    for index,i in enumerate(target):
        plt.subplot(1, 4, index+1) #(rows,col,index)
        image_id_1=data[data[i]==1]['image_id'].iloc[j]
        image=cv2.imread('images/{}.jpg'.format(image_id_1))
        image=cv2.resize(image, (512, 400),interpolation = cv2.INTER_NEAREST)
        list1.append(image)

    for i,col in enumerate(st.columns(4)):
        col.image(list1[i], width=150,caption='%s'%target[i])
        
    return list1

#https://docs.streamlit.io/library/advanced-features/experimental-cache-primitives
@st.experimental_singleton
def load_the_model():
    k_fold_model_1 = tf.keras.models.load_model('model/best_model1_updated.hdf5')
    k_fold_model_2 = tf.keras.models.load_model('model/best_model2_updated.hdf5')
    k_fold_model_3 = tf.keras.models.load_model('model/best_model3_updated.hdf5')
    k_fold_model_4 = tf.keras.models.load_model('model/best_model4_updated.hdf5')
    k_fold_model_5 = tf.keras.models.load_model('model/best_model5_updated.hdf5')

    k_fold_models=[k_fold_model_1,k_fold_model_2,k_fold_model_3,k_fold_model_4,k_fold_model_5]

    return k_fold_models

def predict_func(image):
    target=['healthy','multiple_diseases', 'rust', 'scab']
    loaded_model=load_the_model()

    data_kfold_1=pd.DataFrame()
    for index,model in enumerate(loaded_model):
        pred = model.predict(image[np.newaxis,:,:,:]/255) #predicting on the masked pixel and storing
        pred1=pd.DataFrame(pred,columns=["healthy"+'{}'.format(index), "multiple_diseases"+'{}'.format(index), "rust"+'{}'.format(index),"scab"+'{}'.format(index)])
        pred1.reset_index(drop=True, inplace=True)
        data_kfold_1=pd.concat([data_kfold_1,pred1],axis=1)
    final_df=pd.DataFrame()
    for i,j in enumerate(['healthy','multiple_diseases', 'rust', 'scab']):
        test_avg_column=pd.DataFrame(data_kfold_1.iloc[:,i::4].sum(axis=1)/5,columns=[j])
        final_df =pd.concat([final_df,test_avg_column],axis=1)

    st.image(image,caption="Input Image",width=300)
    st.write('Predicted probabilities for classes')
    st.dataframe(final_df)
    prediction=target[np.argmax(np.array(final_df))]
    new_title = f'<p style="font-family:sans-serif; color:black; font-size: 20px;">Prediction :{prediction}</p>'
    st.markdown(new_title, unsafe_allow_html=True)

if __name__ == "__main__":
    data_train,data_test=main()
    input=st.text_input("Which image wanted to check from the dataset- Train or Test ",'Test',key='str',max_chars=5,help ='test',placeholder='test')
    input_text=st.write(input.strip().capitalize())

    number = st.number_input('Enter_number',key='int',step =1,min_value =0,max_value=500)
    st.write('Input image is :  %s_'%(input.strip().capitalize()), number)

    image=cv2.imread('images/{}_{}.jpg'.format(input.strip().capitalize(),number))
    image=cv2.resize(image, (512, 320),interpolation = cv2.INTER_NEAREST)
    predict_func(image)

    sample_data(data_train)

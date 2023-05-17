#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import Input, layers, Model, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.special import boxcox1p, inv_boxcox
from scipy.stats import boxcox_normmax, boxcox
from keras.regularizers import l2
from mynacos import KOL_MODEL_PATH, LA_PATH
from func import from_mysql_get_all_info
# from tensorflow.python.keras.layers.experimental.preprocessing import PreprocessingLayer

# display(Model is keras.models.Model is tf.keras.models.Model)
np.random.seed(1) # 固定随机种子，使每次运行结果固定


def preprocess(df):
    col = ['fans', 'like_count', 'collect_count', 'comment_count', 'share_count', 'total_count', 'read_count']
    numeric_col = ['fans', 'like_count', 'collect_count', 'comment_count', 'share_count', 'total_count', 'read_count']
    df = df[col]
    df[numeric_col] = df[numeric_col].astype(np.float64)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], -1, inplace=True)
    return df

# method 0: 训练，1：预测
def read_excel(df, method, model_id):

    # cols = df.columns
    # col = list(set(cols).intersection(set(col)))
    # numeric_col = list(set(cols).intersection(set(numeric_col)))
    if method == 'train':
        # df = pd.read_excel(file)
        df = preprocess(df)
        # df['bucket'] = pd.qcut(df['total'], q=150)
        # df['bucket'] = np.floor(np.log10(df['total']))
        # fea_col = ['推荐产品', '粉丝数', '投放阶段', '笔记类型', '赞', '藏', '评', '分享', '互动总量', '阅读量']
        # fea_col = ['推荐产品', '笔记类型', '粉丝数', '赞', '藏', '评', '分享', '互动总量', '阅读量']
        # df = df[fea_col]
        # df.columns = ['product', 'fans', 'step', 'note_type', 'like', 'favorite', 'comment', 'share', 'total', 'view']
        # df.columns = ['product', 'note_type', 'fans', 'like', 'favorite', 'comment', 'share', 'total', 'view']
        # # label = df.pop('view')
        # print(df.isna().sum())
        # print(np.isinf(df).any()[np.isinf(df).any() == True])
        # dd = np.isinf(df['view'])
        # print(np.isinf(df['view']))
        # g = sns.pairplot(df[['fans', 'like', 'favorite', 'comment', 'share', 'total', 'view']], diag_kind="kde")
        # plt.savefig('fig')
        # box - cox变换
        # 计算所有非类别型特征的偏态并排序
        bc_col = ['fans', 'like_count', 'collect_count', 'comment_count', 'share_count', 'total_count']
        skewed_feats = df[bc_col].apply(lambda x: x.skew()).sort_values(ascending=False)
        # 对偏态大于0.5的进行修正，大于0是右偏，小于0是左偏
        high_skew = skewed_feats[abs(skewed_feats) > 0.5]
        skewed_features = high_skew.index
        # 修正
        la_dict = {}
        for feat in skewed_features:
            # 这里是+1是保证数据非负，否则会弹出错误，没有其他含义，不会影响对偏态的修正
            la = boxcox_normmax(df[feat]+1)
            df[feat] = boxcox1p(df[feat], la)
            la_dict[feat] = la
            # df[feat] = boxcox1p(df[feat], la)
        df_la = pd.DataFrame([la_dict]).T
        # df_la.columns = ['la']
        # df_la.to_csv('la.csv')
        # g2 = sns.pairplot(df[['fans', 'like', 'favorite', 'comment', 'share', 'total', 'view_new']], diag_kind="kde")
        # plt.savefig('fig2')
        # plt.show()
        # train_stats = df.describe()
        # train_stats = train_stats.transpose()
        # print(train_stats)
        # dataset_path = keras.utils.get_file("auto-mpg.data",
        #                                     "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
        # column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        #                 'Acceleration', 'Model Year', 'Origin']
        # df = pd.read_csv(dataset_path, names=column_names,
        #                           na_values="?", comment='\t',
        #                           sep=" ", skipinitialspace=True)
        # df = raw_dataset.copy()

        # sns.pairplot(df[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
        return df, df_la
    else:
        # df = pd.DataFrame([file])
        df = preprocess(df)
        # la_path = 'la/la_' + model_id + '.csv'
        la_path = LA_PATH + '/la_' + model_id + '.csv'
        df_la = pd.read_csv(la_path)
        df_la.columns = ['feature', 'la']
        la_dict = df_la.set_index('feature').T.to_dict('list')
        print(la_dict)
        for feat, la in la_dict.items():
            df[feat] = boxcox1p(df[feat], la)
        return df

def df_to_dataset(dataframe, shuffle=True, batch_size=64):
    dataframe = dataframe.copy()
    # X = np.asarray(X).astype(np.float32)
    labels = dataframe.pop('read_count')
    ds = tf.data.Dataset.from_tensors((dict(dataframe), labels))
    # ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict('list'), labels.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        # ds = ds.batch(batch_size)
        # ds = ds.prefetch(batch_size)
    return ds

def df_to_tensor(dataframe, labels):
    # dataframe = dataframe.copy()
    # X = np.asarray(X).astype(np.float32)
    # labels = dataframe.pop('view')
    df = tf.convert_to_tensor(dataframe,dtype=tf.float32)
    la = tf.convert_to_tensor(labels)
    return df, la

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = layers.Normalization(axis=None)
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x,y: x[name])
    # feature_ds = dataset[name]
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x,y: x[name])
    # feature_ds = dataset[name]

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode="one_hot")
    # 类别特征归一化
    # normalizer = layers.Normalization(axis=-1)
    # normalizer.adapt(encoder(index(feature_ds)))

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))

def negative_log_likelihood(y_true, y_pred):
    return -y_pred*tf.math.log(y_true)
    # return -y_pred.log_prob(y_true)

def process_training_data(model_id, is_use):
    # 读取数据
    df = from_mysql_get_all_info('kol', model_id)
    if is_use:
        df_common = from_mysql_get_all_info('kol_default')
        df = pd.concat([df, df_common], ignore_index=True)
        df.drop_duplicates(inplace=True)
    # df = pd.read_excel('all.xlsx')
    df, la = read_excel(df, 'train', model_id)
    if not os.path.exists(LA_PATH):
        print('ok')
        os.makedirs(LA_PATH)
    la_path = LA_PATH + '/la_' + model_id + '.csv'
    # la_path = './la/la_231.csv'
    la.to_csv(la_path)
    # 数据拆分为训练集、验证集和测试集
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    # train.pop('bucket')
    # val.pop('bucket')
    # test.pop('bucket')
    # 输入流水线
    batch_size = 64
    train_ds = df_to_dataset(train, shuffle=True, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    # 选择要使用的列
    all_inputs = []
    encoded_features = []

    # Numeric features.
    # for header in ['fans', 'like', 'favorite', 'comment', 'share', 'total']:
    for header in ['fans', 'like_count', 'collect_count', 'comment_count', 'share_count', 'total_count']:
    # for header in ['Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year']:
        numeric_col = Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)
    #     encoded_features.append(numeric_col)

    # Categorical features encoded as integers.
    # fans_col = tf.keras.Input(shape=(1,), name='fans', dtype='float')
    # buckets = [10000, 50000, 150000, 500000]
    # encoding_layer = layers.Discretization(bin_boundaries=buckets, output_mode='one_hot')
    # encoded_age_col = encoding_layer(fans_col)
    # all_inputs.append(fans_col)
    # encoded_features.append(encoded_age_col)

    # Categorical features encoded as string.
    # categorical_cols = ['product', 'note_type']
    # # categorical_cols = ['Origin']
    # for header in categorical_cols:
    #     categorical_col = Input(shape=(1,), name=header, dtype='string')
    #     encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
    #                                                  max_tokens=9)
    #     encoded_categorical_col = encoding_layer(categorical_col)
    #     all_inputs.append(categorical_col)
    #     encoded_features.append(encoded_categorical_col)

    # 创建、编译并训练模型
    best_k = 0
    min_loss = float('inf')
    best_model = None
    tfd = tfp.distributions
    for k in [5, 10, 20, 30, 50, 64, 100, 128]:  # 网格搜索超参数：神经元数k
        all_features = layers.concatenate(encoded_features, name='f1')
        x = layers.Dense(k, activation='leaky_relu', kernel_initializer='random_uniform', use_bias=True, name='D1')(all_features)
        # x = tf.keras.layers.Dropout(0.1)(x)
        # x = layers.Dense(64, activation='leaky_relu', kernel_initializer='normal')(x)
        x = layers.Dense(k, activation='relu', kernel_initializer='normal', name='D2')(x)
        output = layers.Dense(1, name='Out')(x)
        # output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))(output)
        model = Model(all_inputs, output)
        optimizer = optimizers.RMSprop(0.01)
        model.compile(optimizer=optimizer,
                      loss='mape',
                      metrics=['mse','mae'])
        # model.compile(optimizer='adam',
        #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #               metrics=["accuracy"])
        # rankdir='LR' is used to make the graph horizontal.
        # tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
        # filepath = "best_weights"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)
        # callbacks_list = [checkpoint]
        early_stopping = EarlyStopping(monitor='val_loss', patience=300)
        # model.summary()
        model.fit(train_ds, epochs=5000, validation_data=val_ds, batch_size=64, callbacks=early_stopping, verbose=False)
        # concat_layer_model = Model(inputs=model.input, outputs=model.get_layer('f1').output)
        # concat_output = concat_layer_model.predict(test_ds)
        # dense1_layer_model = Model(inputs=model.get_layer('f1').output, outputs=model.get_layer('D1').output)
        # dense1_output = dense1_layer_model.predict(concat_output)
        # dense2_layer_model = Model(inputs=model.get_layer('D1').output, outputs=model.get_layer('D2').output)
        # dense2_output = dense2_layer_model.predict(dense1_output)
        # out_model = Model(inputs=model.get_layer('D2').output, outputs=model.get_layer('Out').output)
        # final_output = out_model.predict(dense2_output)
        # print(test[0:1])
        # print(test[0:1].values)
        # print(concat_output[0])
        # print(dense1_output[0])
        # print(dense2_output[0])
        # 获得某一层的权重和偏置
        # weight_Dense_1, bias_Dense_1 = model.get_layer('f1').get_weights()
        # print(weight_Dense_1)
        # print(bias_Dense_1)
        # labels = test.pop('view_new')
        # ds = tf.data.Dataset.from_tensor_slices((dict(test), labels))
        # test_ds_new = ds.batch(1)
        # y_pre = np.array([])
        # y_tru = np.array([])
        # for elem in test_ds_new.as_numpy_iterator():
        #     # 注意，这里的model要非训练模式
        #     batch_y_pre = model(elem[0], training=False)
        #     batch_y_tru = elem[1]
        #     y_pre = np.insert(y_pre, len(y_pre), batch_y_pre)
        #     y_tru = np.insert(y_tru, len(y_tru), batch_y_tru)

        loss, mae, mse = model.evaluate(test_ds)
        print(k, " Accuracy", loss, mae, mse)
        if loss < min_loss:
            min_loss = loss
            best_k = k
            best_model = model
    pre = best_model.predict(test_ds)
    best_model.summary()
    model_path = KOL_MODEL_PATH + model_id
    # model_path = 'kol_model_231'
    best_model.save(model_path)
    # i = 0
    # l1, l2 = [], []
    # for index, row in test.iterrows():
    #     y_true = row['view']
    #     x_true = row['view_new']
    #     x = predict[i][0]
    #     y = inv_boxcox(x, la) - 1
    #     l1.append(x)
    #     l2.append(y)
    #     i += 1
    # out = pd.DataFrame({'x':l1, 'y':l2, 'x_true':test['view_new'], 'y_true':test['view']})
    pre_new = pre.T
    out = pd.DataFrame({'pre': pre_new[0], 'true':test['read_count']})
    out['pre_y'] = out['pre'].apply(lambda x:inv_boxcox(x, la)-1)
    out['true_y'] = out['true'].apply(lambda x: inv_boxcox(x, la) - 1)
    out.to_csv('out.csv')


def load_and_predict(model_id, data, method, is_use=0):
    data['total_count'] = data['like_count'] + data['collect_count'] + data['comment_count'] + data['share_count']
    df = data.copy()
    # 新建模型
    # model_path = 'kol_model_' + str(model_id)
    model_path = KOL_MODEL_PATH + model_id
    print(model_path)
    if not os.path.exists(model_path):
        process_training_data(model_id, is_use)
    reconstructed_model = keras.models.load_model(model_path)
    print(reconstructed_model)
    ### 深圳那边写死的模型
    if method == 'kol':
        df = read_excel(df, method, model_id)
        ds = tf.data.Dataset.from_tensors(dict(df))
        pred = reconstructed_model.predict(ds)
        print(pred[0][0])
        return max(pred[0][0], 0), -1
    # 预测
    elif method == 'predict':
        df = read_excel(df, method, model_id)
        ds = tf.data.Dataset.from_tensors(dict(df))
        pred = reconstructed_model.predict(ds)
        data['pred_exposure'] = -1
        data['pred_view'] = pred
        data['pred_view'] = data['pred_view'].apply(lambda x: max(x, 0))
        return data
    # 验证
    else:
        df_val = df[df['status'] == 0]
        data_val = df_val.copy()
        df_train = df[df['status'] != 0]
        df_val, df_train = read_excel(df_val, 'validate', model_id), read_excel(df_train, 'predict', model_id)
        ds_val = df_to_dataset(df_val, shuffle=False)
        if not df_train.empty:
            print('new train samples: ', df_train.shape[0])
            ds_train = df_to_dataset(df_train, shuffle=False)
            reconstructed_model.fit(ds_train)
        # loss, mae, mse = reconstructed_model.evaluate(ds_val)
        # print(k, " Accuracy", loss, mae, mse)
        pred = reconstructed_model.predict(ds_val)
        data_val['pred_exposure'] = -1
        data_val['pred_view'] = pred
        data_val['pred_view'] = data_val['pred_view'].apply(lambda x: max(x, 0))
        return data_val



if __name__ == '__main__':
    # req = {'fans':2662, 'like':16, 'favorite':0, 'comment':14, 'share':0, 'total':30}
#     r_json = {"model_id": 416, "is_use":0, "method":"validate", "data":[
#   {
#     "id": 3882,
#     "status": 0,
#     "recommend_products": "服饰",
#     "blogger": "推广歪歪",
#     "fans": 400,
#     "fans_level": "1k",
#     "put_stage": "3-4月",
#     "release_date": "2020-11-05",
#     "note_type": "视频单品",
#     "report_situation": "1",
#     "post_link": "http://www...",
#     "title": "国风服饰",
#     "like_count": 1010,
#     "collect_count": 389,
#     "comment_count": 25,
#     "share_count": 111,
#     "buy_price": "120.00",
#     "read_count": 0,
#     "exposure": 0,
#     "CPV": "0.00",
#     "CPM": "0.00",
#     "CPE": "0.00",
#     "model_id": 416,
#     "is_delete": 0,
#     "create_time": 1659066711,
#     "update_time": 1659066711,
#     "put_stage2": None,
#     "total_count": 320,
#     "type": 1,
#     "is_common": 0,
#     "predict_read": None,
#     "predict_cpv": None,
#     "index": 0
#   }
# ]}
#     model_id = str(r_json['model_id'])
#     data = r_json['data']
#     method = r_json['method']
#     is_use = r_json['is_use']
#     df = pd.DataFrame(data)
#     res = load_and_predict(model_id, df, method, is_use)
    # load_and_predict(req)
    process_training_data('231', 0)
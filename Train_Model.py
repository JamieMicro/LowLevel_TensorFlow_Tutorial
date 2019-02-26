import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Helpers
def get_one_hot(data):
    # Init numpy array
    if data[0] == 1:
        ret_batch2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 2:
        ret_batch2 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 3:
        ret_batch2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 4:
        ret_batch2 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif data[0] == 13:
        ret_batch2 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif data[0] == 14:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif data[0] == 23:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif data[0] == 24:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0]])
    else:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]])

    for i in range(1, len(data)):
        if data[i] == 1:
            tmparr = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 2:
            tmparr = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 3:
            tmparr = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 4:
            tmparr = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])
        elif data[i] == 13:
            tmparr = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]])
        elif data[i] == 14:
            tmparr = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])
        elif data[i] == 23:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]])
        elif data[i] == 24:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0]])
        else:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]])

        ret_batch2 = np.append(ret_batch2, tmparr, axis=0)

    return ret_batch2
# End Helpers

# Parameters
data_file_path = r'autocar_main_training_data_v30.npy'
model_save_file_path = 'models/v1.0/v1.0'

# Hyper parameters
nn_layers = [64, 64, 9]
learning_rate = .0001
epochs = 5

# Load data
train_data = np.load(data_file_path)

data_rows_count = train_data.shape[0]
data_cols_count = train_data.shape[1]

# Create training data columns headers
feature_columns = []
training_data_cols = []
x_cols = []
y_cols = []

for i in range(0, data_cols_count):
    if i == data_cols_count-1:
        training_data_cols.append('y')
        y_cols.append('y')
        feature_columns.append(tf.feature_column.numeric_column('y'))
    else:
        training_data_cols.append('x' + str(i))
        x_cols.append('x' + str(i))
        feature_columns.append(tf.feature_column.numeric_column('x' + str(i)))

# Create data frame
df_data = pd.DataFrame(data=train_data, index=train_data[:,0], columns=training_data_cols)

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_data[x_cols])
df_normalized = pd.DataFrame(np_scaled)
df_data[x_cols] = df_normalized

# Create train/test split
x_data = df_data.drop('y', axis=1)
y_data = df_data['y']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=99)

# Convert to numpy array and one hot
x_data = x_data.as_matrix()
y_data = y_data.as_matrix()

y_data = get_one_hot(y_data)
y_test = get_one_hot(y_test.as_matrix())

# Initialize parameters
X = tf.placeholder(tf.float32, [None, x_data.shape[1]])
Y = tf.placeholder(tf.float32, [None, nn_layers[len(nn_layers)-1]])

# Randomly initialize
parameters = {}
for l in range(len(nn_layers)):
    if l == 0:
        parameters["W" + str(l + 1)] = tf.Variable(
            tf.random_normal([x_data.shape[1], nn_layers[l]], stddev=0.03), name="W" + str(l + 1))

        parameters["b" + str(l + 1)] = tf.Variable(
            tf.random_normal([nn_layers[l]], stddev=0.03), name="b" + str(l + 1))
    else:
        parameters["W" + str(l + 1)] = tf.Variable(
            tf.random_normal([nn_layers[l - 1], nn_layers[l]], stddev=0.03), name="W" + str(l + 1))

        parameters["b" + str(l + 1)] = tf.Variable(
            tf.random_normal([nn_layers[l]], stddev=0.03), name="b" + str(l + 1))



# Train model
hidden_out_z = {}
hidden_out_a = {}

for l in range(len(nn_layers)-1):
    if l == 0:
        hidden_out_z["Z" + str(l + 1)] = tf.add(tf.matmul(X, parameters["W" + str(l + 1)]), parameters["b" + str(l + 1)])
    else:
        hidden_out_z["Z" + str(l + 1)] = tf.add(tf.matmul(hidden_out_a["A" + str(l)], parameters["W" + str(l + 1)]),
                                                parameters["b" + str(l + 1)])

    # Calculate activation
    hidden_out_a["A" + str(l + 1)] = tf.nn.relu(hidden_out_z["Z" + str(l + 1)])

y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out_a["A" + str(len(nn_layers)-1)], parameters["W" + str(len(nn_layers))]), parameters["b" + str(len(nn_layers))] ))
cross_entropy = tf.nn.l2_loss(y_-Y, name='squared_error_cost')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Setup initilzation operator
init_op = tf.global_variables_initializer()

# Define accuracy assesment
correct_prediction = tf.equal(Y, tf.round(y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predict = tf.round(y_)

# Run the session
model_saver = tf.train.Saver()

with tf.Session() as sess:
    # Init
    sess.run(init_op)

    for epoch in range(epochs):
        avg_cost = 0
        _, c = sess.run([optimizer, cross_entropy], feed_dict={X: x_data, Y: y_data})
        avg_cost += c / len(y_data)
        print("Epoch:", (epoch + 1), "Cost=", "{:.3f}".format(avg_cost))


    print("Accuracy=" + str(sess.run(accuracy, feed_dict={X: x_test, Y: y_test})))
    print("Training Complete")

    # TODO: Below will make predictions given our test input data and return the predicted value (one hot)
    print(sess.run(predict, feed_dict={X: x_test.as_matrix()}))

    # Save Model
    model_saver.save(sess, model_save_file_path)
    print("Model Saved")




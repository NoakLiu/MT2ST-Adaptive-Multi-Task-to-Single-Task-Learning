import matplotlib.pyplot as plt

folder_path = "pics/switch"

def plot2D_accuracy(list1, list2, xname, yname, label, labely):
    name = label
    color1 = "blue"
    color2 = "red"
    x_name = xname # epoch_num
    y_name = yname # accuracy, loss

    plt.figure()
    plt.plot(list1, list2, marker='o', color=color1, label=f'{labely}')
    # plt.plot(list1, list3, marker='*', color=color2, label=f'DFS+GAT (k={k})')
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # Adding a legend
    plt.legend()

    plt.savefig(folder_path+'/{}.png'.format(name))
    plt.show()

def plot2D_loss(list1, list2, xname, yname, label, labely):
    name = label
    color1 = "green"
    color2 = "red"
    x_name = xname # epoch_num
    y_name = yname # accuracy, loss

    plt.figure()
    plt.plot(list1, list2, marker='o', color=color1, label=f'{labely}')
    # plt.plot(list1, list3, marker='*', color=color2, label=f'DFS+GAT (k={k})')
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # Adding a legend
    plt.legend()

    plt.savefig(folder_path+'/{}.png'.format(name))
    plt.show()

def plot2D2_loss(list1, list2, list3, list4,xname, yname, label, labely1, labely2 ):
    print("list1\n",list1)
    print("list2\n", list2)
    print("list3\n", list3)
    print("list4\n", list4)
    name = label
    color1 = "green"
    color2 = "red"
    x_name = xname # epoch_num
    y_name = yname # accuracy, loss

    plt.figure()
    plt.plot(list1, list2, marker='o', color=color1, label=f'{labely1}')
    plt.plot(list3, list4, marker='*', color=color2, label=f'{labely2}')
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # Adding a legend
    plt.legend()

    plt.savefig(folder_path+'/{}.png'.format(name))
    plt.show()

def plot2D2_accuracy(list1, list2, list3, list4,xname, yname, label, labely1, labely2 ):
    name = label
    color1 = "blue"
    color2 = "red"
    x_name = xname # epoch_num
    y_name = yname # accuracy, loss

    plt.figure()
    plt.plot(list1, list2, marker='o', color=color1, label=f'{labely1}')
    plt.plot(list3, list4, marker='*', color=color2, label=f'{labely2}')
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # Adding a legend
    plt.legend()

    plt.savefig(folder_path+'/{}.png'.format(name))
    plt.show()
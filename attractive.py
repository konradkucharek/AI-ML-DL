
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Code to run the decision tree on the Attractive dataset
import dtree

tree = dtree.dtree()
attractive,classes,features = tree.read_data('attractive.data')
t=tree.make_tree(attractive,classes,features)
tree.printTree(t,' ')

print (tree.classifyAll(t,attractive))

for i in range(len(attractive)):
    tree.classify(t,attractive[i])


print "True Classes"
print classes

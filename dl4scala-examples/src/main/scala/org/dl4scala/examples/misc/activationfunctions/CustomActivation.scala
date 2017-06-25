package org.dl4scala.examples.misc.activationfunctions

import org.apache.commons.math3.util
import org.nd4j.linalg.activations.BaseActivationFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Tanh
import org.nd4j.linalg.factory.Nd4j

/**
  * This is an example of how to implement a custom activation function that does not take any learnable parameters
  * Custom activation functions of this case should extend from BaseActivationFunction and implement the methods
  * shown here.
  *
  * The form of the activation function implemented here is from https://arxiv.org/abs/1508.01292
  * "Compact Convolutional Neural Network Cascade for Face Detection" by Kalinovskii I.A. and Spitsyn V.G.
  *
  *      h(x) = 1.7159 tanh(2x/3)
  *
  * Created by endy on 2017/6/25.
  */
class CustomActivation extends BaseActivationFunction{
  override def getActivation(in: INDArray, training: Boolean): INDArray = {
    Nd4j.getExecutioner.execAndReturn(new Tanh(in.muli(2 / 3.0)))
    in.muli(1.7159)
    in
  }

  /**
    * For the backward pass:
    *  Given epsilon, the gradient at every activation node calculate the next set of gradients for the backward pass
    *  Best practice is to modify in place.
    *  Using the terminology,
    *     in -> linear input to the activation node
    *     out    -> the output of the activation node, or in other words h(out) where h is the activation function
    *     epsilon -> the gradient of the loss function with respect to the output of the activation node, d(Loss)/dout
    *         h(in) = out;
    *         d(Loss)/d(in) = d(Loss)/d(out) * d(out)/d(in)
    *                       = epsilon * h'(in)
    * @param in
    * @param epsilon
    * @return
    */
  override def backprop(in: INDArray, epsilon: INDArray): util.Pair[INDArray, INDArray] = {
    val dLdz: INDArray = Nd4j.getExecutioner.execAndReturn(new Tanh(in.muli(2 / 3.0)).derivative)
    dLdz.muli(2 / 3.0)
    dLdz.muli(1.7159)

    //Multiply with epsilon
    dLdz.muli(epsilon)
    new util.Pair(dLdz, null)
  }
}

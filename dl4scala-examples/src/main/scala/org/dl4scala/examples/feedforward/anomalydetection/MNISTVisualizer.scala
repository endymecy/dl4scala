package org.dl4scala.examples.feedforward.anomalydetection

import java.awt.{GridLayout, Image}
import java.awt.image.BufferedImage
import javax.swing.{ImageIcon, JFrame, JLabel, JPanel}

import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.mutable.ArrayBuffer

/**
  * Created by endy on 2017/5/24.
  */
class MNISTVisualizer(imageScale: Double, digits: ArrayBuffer[INDArray], title: String, gridWidth: Int) {
  def this(imageScale: Double, digits: ArrayBuffer[INDArray], title: String) = {
    this(imageScale, digits, title, 5)
  }

  def visualize(): Unit = {
    val frame = new JFrame
    frame.setTitle(title)
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    val panel = new JPanel
    panel.setLayout(new GridLayout(0, gridWidth))
    val list = getComponents
    for (image <- list) {
      panel.add(image)
    }
    frame.add(panel)
    frame.setVisible(true)
    frame.pack()
  }

  def getComponents: ArrayBuffer[JLabel] = {
    val images = new ArrayBuffer[JLabel]()
    for (arr <- digits) {
      val bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
      for(i <- 0 until 784) {
        bi.getRaster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i)).asInstanceOf[Int])
      }
      val orig = new ImageIcon(bi)
      val imageScaled = orig.getImage.getScaledInstance((imageScale * 28).asInstanceOf[Int],
        (imageScale * 28).asInstanceOf[Int], Image.SCALE_REPLICATE)
      val scaled = new ImageIcon(imageScaled)
      images.append(new JLabel(scaled))
    }
    images
  }
}


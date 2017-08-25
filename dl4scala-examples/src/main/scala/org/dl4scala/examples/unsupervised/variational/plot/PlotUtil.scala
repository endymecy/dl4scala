package org.dl4scala.examples.unsupervised.variational.plot

import java.awt.image.BufferedImage
import java.awt._
import javax.swing.event.{ChangeEvent, ChangeListener}
import javax.swing._

import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.{PlotOrientation, XYPlot}
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYDataset, XYSeries, XYSeriesCollection}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.JSlider
import javax.swing.WindowConstants
import java.awt.BorderLayout

import scala.collection.mutable.ArrayBuffer

/**
  * Created by endy on 2017/6/25.
  */
object PlotUtil {

  //Scatterplot util used for CenterLossMnistExample
  def scatterPlot(data: ArrayBuffer[(INDArray,INDArray)], epochCounts: ArrayBuffer[Int], title: String): Unit = {
    var xMin = Double.MaxValue
    var xMax = -Double.MaxValue
    var yMin = Double.MaxValue
    var yMax = -Double.MaxValue

    for (p <- data) {
      val maxes = p._1.max(0)
      val mins = p._1.min(0)
      xMin = Math.min(xMin, mins.getDouble(0))
      xMax = Math.max(xMax, maxes.getDouble(0))
      yMin = Math.min(yMin, mins.getDouble(1))
      yMax = Math.max(yMax, maxes.getDouble(1))
    }

    val plotMin = Math.min(xMin, yMin)
    val plotMax = Math.max(xMax, yMax)

    val panel = new ChartPanel(createChart(data(0)._1, data(0)._2, plotMin, plotMax, title + " (epoch " + epochCounts(0) + ")"))
    val slider = new JSlider(0, epochCounts.size - 1, 0)
    slider.setSnapToTicks(true)

    val f = new JFrame
    slider.addChangeListener(new ChangeListener() {
      private var lastPanel:JPanel = panel

      override def stateChanged(e: ChangeEvent): Unit = {
        val slider: JSlider = e.getSource.asInstanceOf[JSlider]
        val value: Int = slider.getValue
        val panel: JPanel = new ChartPanel(createChart(data(value)._1, data(value)._2, plotMin, plotMax, title + " (epoch " + epochCounts(value) + ")"))
        if (lastPanel != null) {
          f.remove(lastPanel)
        }
        lastPanel = panel
        f.add(panel, BorderLayout.CENTER)
        f.setTitle(title)
        f.revalidate()
      }
    })

    f.setLayout(new BorderLayout)
    f.add(slider, BorderLayout.NORTH)
    f.add(panel, BorderLayout.CENTER)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle(title)

    f.setVisible(true)
  }



  def plotData(xyVsIter: ArrayBuffer[INDArray], labels: INDArray, axisMin: Double, axisMax: Double, plotFrequency: Int): Unit = {
    val panel = new ChartPanel(createChart(xyVsIter(0), labels, axisMin, axisMax))
    val slider = new JSlider(0, xyVsIter.size - 1, 0)
    slider.setSnapToTicks(true)
    val f = new JFrame
    slider.addChangeListener(new ChangeListener() {
      private var lastPanel = panel
      override def stateChanged(e: ChangeEvent): Unit =
      {
        val slider = e.getSource.asInstanceOf[JSlider]
        val value = slider.getValue
        val panel = new ChartPanel(createChart(xyVsIter(value), labels, axisMin, axisMax))
        if (lastPanel != null) f.remove(lastPanel)
        lastPanel = panel
        f.add(panel, BorderLayout.CENTER)
        f.setTitle(getTitle(value, plotFrequency))
        f.revalidate()
      }
    })
    f.setLayout(new BorderLayout)
    f.add(slider, BorderLayout.NORTH)
    f.add(panel, BorderLayout.CENTER)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle(getTitle(0, plotFrequency))
    f.setVisible(true)
  }

  private def getTitle(recordNumber: Int, plotFrequency: Int) =
    "MNIST Test Set - Latent Space Encoding at Training Iteration " + recordNumber * plotFrequency

  // Test data
  private def createDataSet(features: INDArray, labelsOneHot: INDArray): XYDataset  = {
    val nRows = features.rows
    val nClasses = labelsOneHot.columns

    val series = new Array[XYSeries](nClasses)

    (0 until nClasses).foreach(i =>series(i) = new XYSeries(String.valueOf(i)))

    val classIdx = Nd4j.argMax(labelsOneHot, 1)

    (0 until nRows).foreach{i =>
      val idx = classIdx.getInt(i)
      series(idx).add(features.getDouble(i, 0), features.getDouble(i, 1))
    }

    val c = new XYSeriesCollection
    for (s <- series) {
      c.addSeries(s)
    }
    c
  }

  private def createChart(features: INDArray, labels: INDArray, axisMin: Double, axisMax: Double): JFreeChart =
    createChart(features, labels, axisMin, axisMax, "Variational Autoencoder Latent Space - MNIST Test Set")

  private def createChart(features: INDArray, labels: INDArray, axisMin: Double, axisMax: Double, title: String): JFreeChart = {
    val dataset = createDataSet(features, labels)

    val chart = ChartFactory.createScatterPlot(title, "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false)

    val plot = chart.getPlot.asInstanceOf[XYPlot]

    plot.getRenderer.setBaseOutlineStroke(new BasicStroke(0))
    plot.setNoDataMessage("NO DATA")

    plot.setDomainPannable(false)
    plot.setRangePannable(false)
    plot.setDomainZeroBaselineVisible(true)
    plot.setRangeZeroBaselineVisible(true)

    plot.setDomainGridlineStroke(new BasicStroke(0.0f))
    plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f))
    plot.setDomainGridlinePaint(Color.blue)
    plot.setRangeGridlineStroke(new BasicStroke(0.0f))
    plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f))
    plot.setRangeGridlinePaint(Color.blue)

    plot.setDomainMinorGridlinesVisible(true)
    plot.setRangeMinorGridlinesVisible(true)

    val renderer = plot.getRenderer().asInstanceOf[XYLineAndShapeRenderer]

    renderer.setSeriesOutlinePaint(0, Color.black)
    renderer.setUseOutlinePaint(true)
    val domainAxis = plot.getDomainAxis.asInstanceOf[NumberAxis]
    domainAxis.setAutoRangeIncludesZero(false)
    domainAxis.setRange(axisMin, axisMax)

    domainAxis.setTickMarkInsideLength(2.0f)
    domainAxis.setTickMarkOutsideLength(2.0f)

    domainAxis.setMinorTickCount(2)
    domainAxis.setMinorTickMarksVisible(true)

    val rangeAxis = plot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setTickMarkInsideLength(2.0f)
    rangeAxis.setTickMarkOutsideLength(2.0f)
    rangeAxis.setMinorTickCount(2)
    rangeAxis.setMinorTickMarksVisible(true)
    rangeAxis.setRange(axisMin, axisMax)

    chart
  }

  class MNISTLatentSpaceVisualizer(imageScale: Double, digits: ArrayBuffer[INDArray], plotFrequency: Int){
    private val gridWidth = Math.sqrt(digits(0).size(0)).asInstanceOf[Int]

    private def getTitle(recordNumber: Int) =
      "Reconstructions Over Latent Space at Training Iteration " + recordNumber * plotFrequency

    def visualize(): Unit = {
      val frame = new JFrame()
      frame.setTitle(getTitle(0))
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.setLayout(new BorderLayout)

      val panel = new JPanel
      panel.setLayout(new GridLayout(0, gridWidth))

      val slider = new JSlider(0, digits.size - 1, 0)
      slider.addChangeListener(new ChangeListener() {
        override def stateChanged(e: ChangeEvent): Unit = {
          val slider: JSlider = e.getSource.asInstanceOf[JSlider]
          val value: Int = slider.getValue
          panel.removeAll()
          val list = getComponents(value)
          for (image <- list) {
            panel.add(image)
          }
          frame.setTitle(getTitle(value))
        }
      })
      frame.add(slider, BorderLayout.NORTH)

      val list: ArrayBuffer[JLabel] = getComponents(0)
      for (image <- list) {
        panel.add(image)
      }
      frame.add(panel, BorderLayout.CENTER)
      frame.setVisible(true)
      frame.pack()
    }

    private def getComponents(idx: Int): ArrayBuffer[JLabel] = {
      val images = new ArrayBuffer[JLabel]
      val temp = new ArrayBuffer[INDArray]

      (0 until digits(idx).size(0)).foreach(i => temp.append(digits(idx).getRow(i)))


      for (arr <- temp) {
        val bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
        (0 until 784).foreach{i =>
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
}

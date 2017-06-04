package org.dl4scala.examples.recurrent.processnews

import java.awt.event.{ActionEvent, ActionListener}
import java.io.{BufferedReader, File, FileReader}
import javax.swing._

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/4.
  */
class TestNews extends javax.swing.JFrame{

  private val userDirectory = new ClassPathResource("NewsData").getFile.getAbsolutePath + File.separator
  private val WORD_VECTORS_PATH = userDirectory + "NewsWordVector.txt"
  private val wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH))
  private val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)
  private val net = ModelSerializer.restoreMultiLayerNetwork(userDirectory + "NewsModel.net")

  // Variables declaration - do not modify
  private var jButton1: JButton = _
  private var jLabel1: JLabel = _
  private var jLabel2: JLabel = _
  private var jLabel3: JLabel = _
  private var jScrollPane1: JScrollPane = _
  private var jTextArea1: JTextArea = _

  initComponents()

  private def initComponents() = {
    this.setTitle("Predict News Category - KITS")
    jLabel1 = new JLabel
    jScrollPane1 = new JScrollPane
    jTextArea1 = new JTextArea
    jButton1 = new JButton
    jLabel2 = new JLabel
    jLabel3 = new JLabel

    setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE)

    jLabel1.setText("Type News Here")

    jTextArea1.setColumns(20)
    jTextArea1.setRows(5)
    jScrollPane1.setViewportView(jTextArea1)

    jButton1.setText("Check")
    jButton1.addActionListener(new ActionListener() {
      override def actionPerformed(evt: ActionEvent): Unit = {
        jButton1ActionPerformed(evt)
      }
    })

    jLabel2.setText("Category")

    val layout = new GroupLayout(getContentPane)
    getContentPane.setLayout(layout)

    layout.setHorizontalGroup(
      layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
        .addGroup(layout.createSequentialGroup()
          .addContainerGap()
          .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 380, Short.MaxValue)
            .addGroup(layout.createSequentialGroup()
              .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addComponent(jLabel1)
                .addComponent(jButton1))
              .addGap(0, 0, Short.MaxValue))
            .addGroup(layout.createSequentialGroup()
              .addComponent(jLabel2)
              .addGap(18, 18, 18)
              .addComponent(jLabel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MaxValue)))
          .addContainerGap())
    )

    layout.setVerticalGroup(
      layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
        .addGroup(layout.createSequentialGroup()
          .addContainerGap()
          .addComponent(jLabel1)
          .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
          .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 134, javax.swing.GroupLayout.PREFERRED_SIZE)
          .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
          .addComponent(jButton1)
          .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
          .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jLabel3)
            .addComponent(jLabel2))
          .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MaxValue))
    )

    pack()
  }

  private def jButton1ActionPerformed(evt: ActionEvent) = {
    val testNews = prepareTestData(jTextArea1.getText)
    val fet = testNews.getFeatureMatrix
    val predicted = net.output(fet, false)
    val arrsiz = predicted.shape
//    val crimeTotal = 0
//    val politicsTotal = 0
//    val bollywoodTotal = 0
//    val developmentTotal = 0

    val DATA_PATH = userDirectory + "LabelledNews"
    val categories = new File(DATA_PATH + File.separator + "categories.txt")

    var max: Double = 0
    var pos = 0

    (0 until arrsiz(1)).foreach{i =>
      val cur = predicted.getColumn(i).sumNumber.asInstanceOf[Double]
      if (max < cur) {
        max = cur
        pos = i
      }
    }

    try {
      val brCategories = new BufferedReader(new FileReader(categories))
      val labels = new ArrayBuffer[String]()
      Stream.continually(brCategories.readLine()).takeWhile(line => line != null).foreach{line =>
        labels.append(line)
      }
      brCategories.close()
      jLabel3.setText(labels(pos).split(",")(1))
    } catch {
      case e: Exception =>
        TestNews.logger.error("File Exception : " + e.getMessage)
    }
  }

  private def prepareTestData(i_news: String): DataSet = {
    val news = new ArrayBuffer[String](1)
    val category: Array[Int] = new Array[Int](1)
    news.append(i_news)
    val allTokens = new ArrayBuffer[ArrayBuffer[String]](news.size)
    var maxLength = 0
    for(s <- news){
      val tokens = tokenizerFactory.create(s).getTokens
      val tokensFiltered = new ArrayBuffer[String]
      for (t <- tokens.asScala) {
        if (wordVectors.hasWord(t)) tokensFiltered.append(t)
      }
      allTokens.append(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }

    val features = Nd4j.create(news.size, wordVectors.lookupTable.layerSize, maxLength)
    val labels = Nd4j.create(news.size, 4, maxLength) //labels: Crime, Politics, Bollywood, Business&Development

    val featuresMask = Nd4j.zeros(news.size, maxLength)
    val labelsMask = Nd4j.zeros(news.size, maxLength)

    val temp = new Array[Int](2)

    news.indices.foreach{ i =>
      val tokens = allTokens(i)
      temp(0) = i
      for(j <- tokens.indices if j < maxLength) {
        val token = tokens(j)
        val vector = wordVectors.getWordVectorMatrix(token)
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)), vector)
        temp(1) = j
        featuresMask.putScalar(temp, 1.0)
      }
      val idx = category(i)
      val lastIdx = Math.min(tokens.size, maxLength)
      labels.putScalar(Array[Int](i, idx, lastIdx - 1), 1.0)
      labelsMask.putScalar(Array[Int](i, lastIdx - 1), 1.0)
    }
    new DataSet(features, labels, featuresMask, labelsMask)
  }
}

object TestNews extends App {
  val logger: Logger = LoggerFactory.getLogger(classOf[TestNews])

  try{
    var break = false
    for (info <- javax.swing.UIManager.getInstalledLookAndFeels if !break) {
      if ("Nimbus" == info.getName) {
        javax.swing.UIManager.setLookAndFeel(info.getClassName)
        break = true
      }
    }
  }
  catch {
    case ex: ClassNotFoundException =>
      logger.error("ClassNotFoundException: ", ex.getMessage)
    case ex: InstantiationException =>
      logger.error("InstantiationException: ", ex.getMessage)
    case ex: IllegalAccessException =>
      logger.error("IllegalAccessException: ", ex.getMessage)
    case ex: javax.swing.UnsupportedLookAndFeelException =>
      logger.error("UnsupportedLookAndFeelException: ", ex.getMessage)
  }

  val test: TestNews = new TestNews
  test.setVisible(true)
}
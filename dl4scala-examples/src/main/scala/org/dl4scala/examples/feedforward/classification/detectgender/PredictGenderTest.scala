package org.dl4scala.examples.feedforward.classification.detectgender

import java.awt.event.{ActionEvent, ActionListener}

import org.slf4j.{Logger, LoggerFactory}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import javax.swing.JButton
import javax.swing.JTextField
import javax.swing.JLabel
import javax.swing.JDialog
import javax.swing.WindowConstants

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j


/**
  * Created by endy on 2017/5/17.
  */
class PredictGenderTest extends Runnable{
  val log: Logger = LoggerFactory.getLogger(classOf[PredictGenderTest])

  private var jd: JDialog = _
  private var jtf: JTextField = _
  private var jlbl: JLabel = _
  private var possibleCharacters: String = _
  private var gender: JLabel = _
  private var filePath: String = _
  private var btnNext: JButton = _
  private var genderLabel: JLabel = _
  private var model: MultiLayerNetwork = _

  def prepareInterface(): Unit = {
    this.jd = new JDialog
    this.jd.getContentPane.setLayout(null)
    this.jd.setBounds(100, 100, 300, 250)
    this.jd.setLocationRelativeTo(null)
    this.jd.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE)
    this.jd.setTitle("Predict Gender By Name")
    this.jd.setVisible(true)

    this.jlbl = new JLabel
    this.jlbl.setBounds(5, 10, 100, 20)
    this.jlbl.setText("Enter Name : ")
    this.jd.add(jlbl)

    this.jtf = new JTextField
    this.jtf.setBounds(105, 10, 150, 20)
    this.jd.add(jtf)

    this.genderLabel = new JLabel
    this.genderLabel.setBounds(5, 12, 70, 170)
    this.genderLabel.setText("Gender : ")
    this.jd.add(genderLabel)

    this.gender = new JLabel
    this.gender.setBounds(75, 12, 75, 170)
    this.jd.add(gender)

    this.btnNext = new JButton
    this.btnNext.setBounds(5, 150, 150, 20)
    this.btnNext.setText("Predict")

    this.btnNext.addActionListener(new ActionListener() {
      override def actionPerformed(e: ActionEvent): Unit = {
        if (!jtf.getText().isEmpty) {
          val binaryData = getBinaryString(jtf.getText.toLowerCase)

          val arr = binaryData.split(",")
          val db = new Array[Int](arr.length)
          val features = Nd4j.zeros(1, 220)

          for(i <- 0 until arr.length){
            features.putScalar(Array[Int](0, i), arr(i).toInt)
          }
          val predicted = model.output(features)
          if (predicted.getDouble(0) > predicted.getDouble(1)) gender.setText("Female")
          else if (predicted.getDouble(0) < predicted.getDouble(1)) gender.setText("Male")
          else gender.setText("Both male and female can have this name")
        }
        else gender.setText("Enter name please..")
      }
    })

    this.jd.add(this.btnNext)
  }

  private def getBinaryString(name: String): String = {
    var binaryString = ""
    for (j <- 0 until name.length){
      val fs = pad(Integer.toBinaryString(possibleCharacters.indexOf(name.charAt(j))), 5)
      binaryString = binaryString + fs
    }

    var diff = 0
    if (name.length < 44){
      diff = 44 - name.length
      for (_ <- 0 until diff){
        binaryString = binaryString + "00000"
      }
    }

    var tempStr = ""
    for(i <- 0 until binaryString.length){
      tempStr = tempStr + binaryString.charAt(i) + ","
    }

    tempStr
  }

  private def pad(string: String, total_length: Int): String = {
    var str = string

    val diff = if (total_length > string.length) total_length - string.length else 0

    (0 until diff).foreach(_ => str = "0" + str)

    str
  }

  override def run(): Unit = {
    try
    {
      this.filePath = new ClassPathResource("/PredictGender/Data/").getFile.getPath
      this.possibleCharacters = " abcdefghijklmnopqrstuvwxyz"
      this.model = ModelSerializer.restoreMultiLayerNetwork(this.filePath + "/PredictGender1.net")
    } catch {
      case e: Exception =>
        log.error("Exception : " + e.getMessage)
    }
  }
}

object PredictGenderTest extends App {
  val pgt: PredictGenderTest = new PredictGenderTest
  val t: Thread = new Thread(pgt)
  t.start()
  pgt.prepareInterface()
}

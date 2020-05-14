package org.ucf.ml
package utils

import java.io.File
import com.typesafe.config.ConfigFactory

class PropertiesLoader(configPath:String = "src/main/resources/application.conf") {

  val parsedConfig = ConfigFactory.parseFile(new File(configPath))

  private val conf = ConfigFactory.load(parsedConfig)


  val getLogLevel = conf.getString("LogLevel")

  def getIdiomsPath = conf.getString("IdiomsPath")
  def getRawBuggyFilesDir = conf.getString("RawBuggyFilesDir")
  def getRawFixedFilesDir = conf.getString("RawFixedFilesDir")
  def getOutputBuggyDir = conf.getString("OutputBuggyDir")
  def getOutputFixedDir = conf.getString("OutputFixedDir")
}
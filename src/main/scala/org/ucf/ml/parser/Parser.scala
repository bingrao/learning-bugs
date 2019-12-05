package org.ucf.ml
package parser

import java.io.File

import com.github.javaparser.JavaParser

import scala.collection.mutable.ListBuffer

class Parser extends plaintext.SourceCodeAnalyzer with Visitor {
  def parseFile(filePath:String) = {
    // Parse some code
    val reader = readSourceCode(filePath)
    val src = "public class DummyClass {" + removeCommentsAndAnnotations(reader) + "}"
    val cu = JavaParser.parse(src)
    val methodNames = ListBuffer[String]()
    val methodNameCollector = new MethodNameCollector
    methodNameCollector.visit(cu,methodNames)
    methodNames.foreach(println _)
  }
}
package org.ucf.ml

object TestUtils extends parser.JavaParser{
  // Load data idioms
  private val idioms = readIdioms()
  val ctx = new Context(idioms)

  def get_abstract_code(sourcePath:String, granularity:Value, isFile:Boolean = true) = {

    ctx.setCurrentMode(SOURCE)
    ctx.setNewLine(true)
    ctx.setIsAbstract(false)
    val cu = getComplationUnit(sourcePath, granularity, isFile)

    printAST(outPath="log/test.Yaml", cu = cu, format = "ymal")

    addPositionWithGenCode(ctx, cu)
    ctx.clear
    ctx.get_buggy_abstract.toString
  }

}


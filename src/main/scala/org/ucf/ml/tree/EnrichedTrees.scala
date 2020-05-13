package org.ucf.ml
package tree

import scala.collection.JavaConversions._
import com.github.javaparser.ast.body._
import com.github.javaparser.ast._
import utils.Context


trait EnrichedTrees extends _Statement{

  implicit class addPosition(node:Node) {
    def getPosition(ctx: utils.Context) = ctx.getNewPosition
  }

  implicit class genCompilationUnit(node:CompilationUnit) {
    def genCode(ctx:Context):String = {
      // 1. package declaration
      val package_decl = node.getPackageDeclaration
      if (package_decl.isPresent) package_decl.get().genCode(ctx)

      // 2. Import Statements
      node.getImports.toList.foreach(impl => impl.genCode(ctx))

      // 3. A list of defined types, such as Class, Interface, Enum, Annotation ...
      node.getTypes.toList.foreach(typeDecl => typeDecl.genCode(ctx))

      EMPTY_STRING
    }
  }

  implicit class genPackageDeclaration(node: PackageDeclaration) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getName.toString)
      EMPTY_STRING
    }
  }

  implicit class genImportDeclaration(node:ImportDeclaration) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getName.toString)
      EMPTY_STRING
    }
  }

  implicit class genTypeDeclaration(node:TypeDeclaration[_]) {
    def genCode(ctx:Context):String = {
      node match {
        /**
         *  TypeDeclaration
         *     -- EnumDeclaration
         *     -- AnnotationDeclaration
         *     -- ClassOrInterfaceDeclaration
         */
        case n: EnumDeclaration => {n.genCode(ctx)}
        case n: AnnotationDeclaration => {n.genCode(ctx)}
        case n: ClassOrInterfaceDeclaration => {n.genCode(ctx)}
      }
      EMPTY_STRING
    }
  }

  implicit class genEnumDeclaration(node:EnumDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genAnnotationDeclaration(node:AnnotationDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genClassOrInterfaceDeclaration(node:ClassOrInterfaceDeclaration) {
    def genCode(ctx:Context):String = {
      // 1. Class Modifiers, such as public/private
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))
      // 2. Class Name
      node.getName.genCode(ctx)

      ctx.append("{")
      ctx.appendNewLine()

      // 3. Class Members
      val members = node.getMembers.toList
      members.foreach(bodyDecl => bodyDecl.genCode(ctx))
      ctx.append("}")
      ctx.appendNewLine()
      EMPTY_STRING
    }
  }

  implicit class genBodyDeclaration(node:BodyDeclaration[_]){
    def genCode(ctx:Context):String = {
      node match {
        case n: InitializerDeclaration => {n.genCode(ctx)}
        case n: FieldDeclaration => {n.genCode(ctx)}
        case n: TypeDeclaration[_] => n.genCode(ctx)
        case n: EnumConstantDeclaration => {n.genCode(ctx)}
        case n: AnnotationMemberDeclaration => {n.genCode(ctx)}
        case n: CallableDeclaration[_] => {n.genCode(ctx)}

      }
      EMPTY_STRING
    }
  }

  implicit class genInitializerDeclaration(node:InitializerDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genFieldDeclaration(node:FieldDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genEnumConstantDeclaration(node:EnumConstantDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genAnnotationMemberDeclaration(node:AnnotationMemberDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genCallableDeclaration(node:CallableDeclaration[_]){
    def genCode(ctx:Context):String = {
      node match {
        /**
         * CallableDeclaration
         *    -- ConstructorDeclaration
         *    -- MethodDeclaration
         */
        case n: ConstructorDeclaration => {n.genCode(ctx)}
        case n: MethodDeclaration => {n.genCode(ctx)}
      }
      EMPTY_STRING
    }
  }

  implicit class genConstructorDeclaration(node:ConstructorDeclaration) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genMethodDeclaration(node:MethodDeclaration) {
    def genCode(ctx:Context):String = {
      /*modifiers, such as public*/
      val modifiers = node.getModifiers.toList
      modifiers.foreach(modifier => modifier.genCode(ctx))

      /*Method return type, such as void, int, string*/
      node.getType.genCode(ctx)

      /*Method name, such as hello*/
      val value = ctx.method_maps.getNewContent(node.getName.toString())
      ctx.append(value)

      /*formal paramters*/
      ctx.append("(")
      val parameters = node.getParameters.toList
      parameters.foreach(p => {
        p.genCode(ctx)
        if (p != parameters.last) ctx.append(",")
      })
      ctx.append(")")

      /*Method Body*/
      val body = node.getBody
      if (body.isPresent) body.get().genCode(ctx)

      EMPTY_STRING
    }
  }

}

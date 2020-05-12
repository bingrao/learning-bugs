package org.ucf.ml;
public class JavaApp {
    public static void hello(String input) {
        System.out.println(input);
    }

    public static void main(String[] args) {
        String input = "output";
        com.github.javaparser.ast.CompilationUnit cu;
        System.out.println("This will be printed" + input);

        hello(input);
    }

}

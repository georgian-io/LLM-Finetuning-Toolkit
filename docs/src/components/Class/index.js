import React from "react";
import clsx from "clsx";
import { MDXProvider } from "@mdx-js/react";
import SyntaxHighlighter from "prism-react-renderer";

// Your CSS module, assuming you have one for these components
import styles from "./styles.module.css";

export const ClassComponent = ({
  className,
  initParams,
  sourceUrl,
  isClass = true,
  children,
}) => {
  return (
    <div className={clsx(styles.classContainer)}>
      <div className={clsx(styles.classHeader)}>
        <div className={clsx(styles.classInfo)}>
          <span className={clsx(styles.classType)}>
            {isClass ? "class" : "function"}
          </span>
          <span>
            {sourceUrl && (
              <a
                className={clsx(styles.sourceLink)}
                href={sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontFamily: "monospace" }}
              >
                {"< source >"}
              </a>
            )}
          </span>
        </div>
        <h2 className={clsx(styles.className)}>{className}</h2>
      </div>
      <div className={clsx(styles.initParams)}>{initParams}</div>
      {children}
    </div>
  );
};

export const MethodComponent = ({ methodName, methodSignature, children }) => {
  return (
    <div className={clsx(styles.methodContainer)}>
      <h2 className={clsx(styles.methodName)}>{methodName}</h2>
      <p className={clsx(styles.methodSignature)}>{methodSignature}</p>
      <div>{children}</div>
    </div>
  );
};

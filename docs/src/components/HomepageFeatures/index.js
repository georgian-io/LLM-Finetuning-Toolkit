import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "Streamlined Workflow for Data Scientists",
    Svg: require("@site/static/img/undraw_scientist.svg").default,
    description: (
      <>
        This toolkit is designed specifically for data scientists and
        researchers, providing a streamlined and intuitive workflow for running
        experiments efficiently. It abstracts away technical complexities,
        allowing users to focus on their research and quickly iterate on ideas.
      </>
    ),
  },
  {
    title: "Flexible and Modular Architecture",
    Svg: require("@site/static/img/undraw_design_components.svg").default,
    description: (
      <>
        The toolkit's modular architecture enables easy customization and
        extension of functionality. Users can incorporate custom data formats,
        experiment with different finetuning techniques, and define their own
        evaluation metrics, adapting the toolkit to their specific needs.
      </>
    ),
  },
  {
    title: "Comprehensive Model Quality Assurance",
    Svg: require("@site/static/img/undraw_qa.svg").default,
    description: (
      <>
        The toolkit emphasizes quality assurance, providing a comprehensive
        suite of QA tests to assess model performance. Users can evaluate
        accuracy, fluency, diversity, and consistency, and extend the tests to
        define custom evaluation criteria specific to their domain or task.
      </>
    ),
  },
  {
    title: "Reproducibility and Experiment Tracking",
    Svg: require("@site/static/img/undraw_notes.svg").default,
    description: (
      <>
        Designed with reproducibility in mind, the toolkit includes built-in
        mechanisms for tracking and managing experiments. It generates unique
        experiment IDs and maintains a structured directory hierarchy, ensuring
        proper recording and storage of configurations, hyperparameters, and
        results for easy reproduction and comparison.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx("col col--3")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

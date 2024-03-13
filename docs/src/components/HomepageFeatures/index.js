import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "Streamlined Workflow for Data Scientists",
    Svg: require("@site/static/img/undraw_scientist.svg").default,
    description: (
      <>
        This toolkit provides data scientists with an intuitive workflow for
        running experiments efficiently, abstracting away technical complexities
        and allowing users to focus on their research.
      </>
    ),
  },
  {
    title: "Flexible and Modular Architecture",
    Svg: require("@site/static/img/undraw_design_components.svg").default,
    description: (
      <>
        The modular architecture enables easy customization and extension of
        functionality, allowing users to incorporate custom data formats,
        experiment with different techniques, and define their own evaluation
        metrics.
      </>
    ),
  },
  {
    title: "Comprehensive Model Quality Assurance",
    Svg: require("@site/static/img/undraw_qa.svg").default,
    description: (
      <>
        The toolkit emphasizes quality assurance, providing a suite of tests to
        assess model performance across accuracy, fluency, diversity, and
        consistency. Users can extend the tests to define custom evaluation
        criteria.
      </>
    ),
  },
  {
    title: "Reproducibility and Experiment Tracking",
    Svg: require("@site/static/img/undraw_notes.svg").default,
    description: (
      <>
        Designed for reproducibility, the toolkit includes mechanisms for
        tracking experiments, generating unique IDs, and maintaining a
        structured directory hierarchy for easy reproduction and comparison of
        results.
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

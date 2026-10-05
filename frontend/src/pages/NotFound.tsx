import { Link } from "react-router-dom";

export function NotFound() {
  return (
    <section>
      <h1 className="mb-2 text-xl font-semibold">页面不存在</h1>
      <Link className="text-sm text-sky-600 hover:underline dark:text-sky-400" to="/library">
        回到素材库
      </Link>
    </section>
  );
}

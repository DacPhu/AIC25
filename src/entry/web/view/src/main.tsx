import * as React from "react";
import * as ReactDOM from "react-dom/client";

import {
  createRoutesFromElements,
  createBrowserRouter,
  Route,
  RouterProvider,
} from "react-router-dom";

import Root, { loader as RootLoader } from "@/routes/Root";
import Search, { loader as SearchLoader } from "@/routes/Search";
import { loader as SearchSimilarLoader } from "@/routes/SearchSimilar";
import {
  action as AnswerAction,
  loader as AnswerLoader,
} from "@/routes/answers";
import { action as AnswerDeleteAction } from "@/routes/answersdelete";
import { action as AnswerEditAction } from "@/routes/answeredit";

import "./index.css";
import { initPerformanceMonitoring } from "@/utils/performance";

if (import.meta.env.PROD) {
  initPerformanceMonitoring();
}

const router = createBrowserRouter(
  createRoutesFromElements([
    <Route path="/" element={<Root />} loader={RootLoader}>
      <Route path="search" element={<Search />} loader={SearchLoader} />
      <Route path="similar" element={<Search />} loader={SearchSimilarLoader} />
    </Route>,
    <Route path="answers" action={AnswerAction} loader={AnswerLoader}>
      <Route path=":answerId">
        <Route path="delete" action={AnswerDeleteAction} />
        <Route path="edit" action={AnswerEditAction} />
      </Route>
    </Route>,
  ]),
);

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element not found");

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
